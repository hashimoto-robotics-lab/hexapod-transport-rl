"""Change only the action set to stop + four world directions."""

from hexapod_transport_rl.student import CylinderPushTask, DiscreteDirections

TASK = CylinderPushTask(actions=DiscreteDirections(count=4, include_stop=True))

# Other action experiments (run separately):
# TASK = CylinderPushTask(actions=DiscreteDirections(count=16))
# from hexapod_transport_rl.student import ContinuousVelocity
# TASK = CylinderPushTask(actions=ContinuousVelocity(frame="world", speed=.15))
