"""Private GitHub checkout for Colab; copied as readable code into the notebook."""

import getpass
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


def checkout_repository(repository, ref, destination, project_subdir=""):
    """Fetch once, record the commit, and preserve student edits on later calls."""
    repository = repository.strip().removeprefix("https://github.com/")
    repository = repository.rstrip("/").removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("リポジトリは所有者名/リポジトリ名で指定してください。")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", ref):
        raise ValueError("版にはブランチ名・タグ名・コミットSHAを指定してください。")
    destination = Path(destination).resolve()
    project = (destination / project_subdir).resolve()
    if not project.is_relative_to(destination):
        raise ValueError("教材の場所はリポジトリ内の相対パスで指定してください。")
    url = f"https://github.com/{repository}.git"
    expected = dict(repository=repository, ref=ref)
    marker = destination / ".colab_checkout.json"

    def git(*arguments, cwd, env=None):
        return subprocess.run(
            ["git", "-c", "credential.helper=", *arguments],
            cwd=cwd,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise RuntimeError(
                "取得先に別の教材があります。新しいランタイムで取得してください。"
            )
        if git("config", "--get", "remote.origin.url", cwd=destination) != url:
            raise RuntimeError("取得先のリポジトリが異なります。")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            temporary = Path(temporary)
            checkout = temporary / "repository"
            checkout.mkdir()
            git("init", "--quiet", cwd=checkout)
            git("remote", "add", "origin", url, cwd=checkout)
            askpass = temporary / "askpass.sh"
            askpass.write_text(
                '#!/bin/sh\ncase "$1" in\n'
                '  *Username*) printf "%s\\n" "x-access-token" ;;\n'
                '  *) printf "%s\\n" "$HEXAPOD_GIT_TOKEN" ;;\nesac\n'
            )
            askpass.chmod(0o700)
            # The token exists only in this fetch process's environment.
            env = {
                **os.environ,
                "GIT_ASKPASS": str(askpass),
                "GIT_TERMINAL_PROMPT": "0",
                "HEXAPOD_GIT_TOKEN": getpass.getpass(
                    "GitHubのアクセストークン（非表示）: "
                ),
            }
            try:
                git(
                    "fetch",
                    "--quiet",
                    "--depth",
                    "1",
                    "origin",
                    ref,
                    cwd=checkout,
                    env=env,
                )
                git("checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=checkout)
            except subprocess.CalledProcessError:
                raise RuntimeError(
                    "教材を取得できません。招待の承諾、トークンの権限、版を確認してください。"
                ) from None
            finally:
                env.clear()
            (checkout / marker.name).write_text(json.dumps(expected) + "\n")
            checkout.rename(destination)

    if not (project / "pyproject.toml").is_file():
        raise RuntimeError("教材が見つかりません。PROJECT_SUBDIRを確認してください。")
    commit = git("rev-parse", "HEAD", cwd=destination)
    return project, commit
