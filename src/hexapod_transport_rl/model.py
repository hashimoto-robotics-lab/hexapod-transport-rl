"""Compose independent articulated robots; collisions transmit all cargo forces."""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .config import JOINT_NAMES, MODEL_RELATIVE_PATH, PHYSICS_DT, PushConfig
from .motor import add_motor


def build_model(asset_root: Path, cfg: PushConfig) -> mujoco.MjModel:
    """Compose source robot copies, physical cargo, goal marker and DC motors."""
    root, world, robot_template = _prepare_scene(asset_root)
    all_bits = _add_robots_and_floor(root, world, robot_template, cfg.num_robots)
    cargo = _add_cargo(world, cfg, all_bits)
    _add_goal_marker(world, cargo)
    return _compile_model(root, cfg.num_robots)


def _compile_model(root: ET.Element, num_robots: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))
    for i in range(num_robots):
        for name in JOINT_NAMES:
            add_motor(spec, f"r{i}/{name}")
    return spec.compile()


def _prepare_scene(asset_root: Path) -> tuple[ET.Element, ET.Element, ET.Element]:
    """Copy source geometry and reproduce the walking model's mass and joints."""
    source = asset_root / MODEL_RELATIVE_PATH
    root = ET.parse(source).getroot()
    root.set("model", "hexapod_cooperative_transport")
    root.find("compiler").set("meshdir", str(source.parent / "meshes"))
    for tag in ("sensor", "actuator", "keyframe", "contact"):
        for element in root.findall(tag):
            root.remove(element)
    ET.SubElement(
        root,
        "option",
        timestep=str(PHYSICS_DT),
        integrator="implicitfast",
        solver="Newton",
        iterations="30",
        ls_iterations="20",
        cone="pyramidal",
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="960", offheight="720")
    collision = root.find("default/default/default[@class='collision']/geom")
    collision.set("friction", ".8 .005 .0001")
    collision.set("solref", ".01 1")
    collision.set("condim", "3")
    world = root.find("worldbody")
    original = copy.deepcopy(world.find("body"))
    inertials = list(original.iter("inertial"))
    factor = 3.0 / sum((float(e.get("mass")) for e in inertials))
    for element in inertials:
        element.set("mass", str(float(element.get("mass")) * factor))
        for key in ("fullinertia", "diaginertia"):
            if key in element.attrib:
                element.set(
                    key,
                    " ".join(
                        (
                            str(v * factor)
                            for v in np.fromstring(element.get(key), sep=" ")
                        )
                    ),
                )
    for joint in original.iter("joint"):
        joint.set("armature", ".025")
        joint.set("damping", ".05")
        joint.set("frictionloss", ".05")
    world.clear()
    ET.SubElement(world, "light", pos="0 -2 5", dir="0 0 -1", diffuse=".8 .8 .8")
    return (root, world, original)


def _add_robots_and_floor(
    root: ET.Element, world: ET.Element, robot_template: ET.Element, num_robots: int
) -> int:
    """Namespace each robot and enable external contacts, excluding self-collision.

    Collision bits: floor=1, robot i=1<<(i+1), cargo=1<<(N+1).
    The returned mask reserves one additional bit for optional cargo.
    """
    all_bits = (1 << num_robots + 2) - 1
    _add_floor_material(root)
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 .1",
        contype="1",
        conaffinity=str(all_bits),
        friction=".8 .005 .0001",
        material="push_floor_grid",
    )
    sensors = ET.SubElement(root, "sensor")
    for i in range(num_robots):
        body = copy.deepcopy(robot_template)
        bit = 1 << i + 1
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name", f"r{i}/" + element.get("name"))
            if element.tag == "geom":
                if "collision" in element.get("name", ""):
                    element.set("contype", str(bit))
                    element.set("conaffinity", str(all_bits ^ bit))
        world.append(body)
        ET.SubElement(sensors, "gyro", name=f"r{i}/imu_ang_vel", site=f"r{i}/imu")
    return all_bits


def _add_floor_material(root: ET.Element) -> None:
    """Draw 0.5 m grid lines with a texture; no extra collision geometry."""
    assets = root.find("asset")
    if assets is None:
        assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="push_floor_grid_texture",
        type="2d",
        builtin="flat",
        rgb1=".78 .81 .84",
        mark="edge",
        markrgb=".35 .39 .43",
        width="128",
        height="128",
    )
    ET.SubElement(
        assets,
        "material",
        name="push_floor_grid",
        texture="push_floor_grid_texture",
        texuniform="true",
        texrepeat="2 2",
        rgba="1 1 1 1",
        specular=".1",
        shininess=".1",
    )


def _add_cargo(world: ET.Element, cfg: PushConfig, all_bits: int) -> ET.Element:
    """Build a box, a cylinder, or the two non-overlapping pieces of a T."""
    half_height = cfg.cargo_height / 2
    cargo = ET.SubElement(world, "body", name="cargo", pos=f"0 0 {half_height}")
    ET.SubElement(cargo, "freejoint", name="cargo_free")
    cargo_bit = 1 << cfg.num_robots + 1
    common = dict(
        type="box",
        contype=str(cargo_bit),
        conaffinity=str(all_bits),
        friction=f"{cfg.cargo_friction} .005 .0001",
        priority="1",
        rgba=".9 .66 .15 1",
    )
    ET.SubElement(
        cargo,
        "geom",
        name="cargo_cylinder",
        size=f"{cfg.cargo_radius} {half_height}",
        mass=str(cfg.cargo_mass),
        **{**common, "type": "cylinder"},
    )
    return cargo


def _add_goal_marker(world: ET.Element, cargo: ET.Element) -> None:
    """Show the full desired footprint with collision-free translucent geoms."""
    target = ET.SubElement(world, "body", name="goal")
    for geom in cargo.findall("geom"):
        marker = copy.deepcopy(geom)
        marker.set("name", "goal_" + geom.get("name"))
        marker.set("contype", "0")
        marker.set("conaffinity", "0")
        marker.set("rgba", ".15 .75 .35 .22")
        marker.set("mass", "0")
        target.append(marker)
