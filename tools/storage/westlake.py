#!/usr/bin/env python3
"""Connect to the existing Westlake project over SSH/rsync (no stored credentials)."""
import argparse
from pathlib import PurePosixPath
import shlex
import subprocess

ROOT = "/storage/yukaichengLab/wangxin/test_eiloeo"


def relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or ":" in value:
        raise argparse.ArgumentTypeError("Use a path relative to the Westlake project root")
    return str(PurePosixPath(ROOT)/path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="gpu-wangxin")
    p.add_argument("action", choices=["status", "list", "shell", "pull", "push"])
    p.add_argument("paths", nargs="*")
    a = p.parse_args()
    if a.host.startswith("-"):
        p.error("Invalid SSH host")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", a.host]
    if a.action == "status":
        command = "hostname; ls -ld " + shlex.quote(ROOT) + "; df -h " + shlex.quote(ROOT)
        subprocess.run(ssh + [command], check=True)
    elif a.action == "list":
        target = relative_path(a.paths[0] if a.paths else ".")
        subprocess.run(ssh + ["ls -lah -- " + shlex.quote(target)], check=True)
    elif a.action == "shell":
        subprocess.run(ssh[:-1] + ["-t", a.host, "cd " + shlex.quote(ROOT) + " && exec bash -l"], check=True)
    else:
        if len(a.paths) != 2:
            p.error("pull: REMOTE_RELATIVE LOCAL; push: LOCAL REMOTE_RELATIVE")
        index = 0 if a.action == "pull" else 1
        remote = a.host + ":" + shlex.quote(relative_path(a.paths[index]))
        pair = [remote, a.paths[1]] if a.action == "pull" else [a.paths[0], remote]
        # Deliberately additive: never delete files on either endpoint.
        subprocess.run(["rsync", "-av", "--partial", "-e", "ssh -o BatchMode=yes -o ConnectTimeout=15", "--"] + pair, check=True)


if __name__ == "__main__":
    main()
