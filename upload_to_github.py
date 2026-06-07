"""
upload_to_github.py
===================
Helper script to stage, commit, and push all project files to GitHub.

Usage:
    python upload_to_github.py
    python upload_to_github.py --message "Add improved GCS downloader with tas support"
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)
    if check and result.returncode != 0:
        print(f"  ERROR: command failed with exit code {result.returncode}", flush=True)
        sys.exit(result.returncode)
    return result


def main(message: str):
    repo_root = Path(__file__).resolve().parent
    print(f"Repository root: {repo_root}", flush=True)

    print("\n[1] Checking git status ...")
    run(["git", "-C", str(repo_root), "status", "--short"])

    print("\n[2] Staging all changes ...")
    run(["git", "-C", str(repo_root), "add", "."])

    print("\n[3] Committing ...")
    result = run(
        ["git", "-C", str(repo_root), "commit", "-m", message],
        check=False)
    if result.returncode != 0 and "nothing to commit" in result.stdout + result.stderr:
        print("  Nothing new to commit — working tree clean.", flush=True)
    elif result.returncode != 0:
        print(f"  Commit failed.", flush=True)
        sys.exit(result.returncode)

    print("\n[4] Pushing to origin/main ...")
    run(["git", "-C", str(repo_root), "push", "origin", "main"])

    print("\n✓ Done. Repository is up to date.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stage, commit, and push to GitHub")
    p.add_argument(
        "--message", "-m",
        default="Update project files",
        help="Commit message (default: 'Update project files')")
    a = p.parse_args()
    main(a.message)
