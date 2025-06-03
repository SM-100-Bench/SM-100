#!/usr/bin/env python3
import argparse
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple, Optional

from git import Repo  # type: ignore
import requests
from tqdm import tqdm  # progress bar

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    github_get,
    _parse_patch_line_map,
    _merge_line_maps,
)


def build_docker_image(dockerfile_path: Path, tag: str) -> bool:
    """
    Build a Docker image from the specified Dockerfile.

    Args:
        dockerfile_path: Path to the Dockerfile
        tag: Tag to apply to the built image

    Returns:
        True if the build was successful, False otherwise
    """
    logging.info(f"Building Docker image from {dockerfile_path} with tag {tag}")
    try:
        result = subprocess.run(
            [
                "docker",
                "build",
                "-t",
                tag,
                "-f",
                str(dockerfile_path),
                str(dockerfile_path.parent),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logging.debug(f"Docker build output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Docker build failed: {e.stderr}")
        return False


def run_check_script(
    script_path: Path, repo_path: Path, output_file: Optional[Path] = None
) -> int:
    """
    Run a check script with the repository path.

    Args:
        script_path: Path to the check script
        repo_path: Path to the repository
        output_file: Optional path to save command output

    Returns:
        The exit code of the script
    """
    logging.info(f"Running check script {script_path} with repo at {repo_path}")

    try:
        # Make sure the script is executable
        script_path.chmod(0o755)

        if output_file:
            with open(output_file, "w") as f:
                result = subprocess.run(
                    [str(script_path.absolute())],
                    cwd=str(repo_path),
                    check=False,  # Don't raise exception on non-zero exit
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            result = subprocess.run(
                [str(script_path.absolute())],
                cwd=str(repo_path),
                check=False,  # Don't raise exception on non-zero exit
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logging.debug(f"Check script output: {result.stdout}")

        return result.returncode
    except Exception as e:
        logging.error(f"Check script run failed: {e}")
        return -1


def run_docker_container(
    tag: str, repo_path: Path, output_file: Optional[Path] = None
) -> int:
    """
    Run a Docker container with the specified tag, mounting the repo to /repo.

    Args:
        tag: Docker image tag to run
        repo_path: Path to the repository to mount
        output_file: Optional path to save command output

    Returns:
        The exit code of the container
    """
    logging.info(f"Running Docker container {tag} with repo mounted at {repo_path}")

    cmd = [
        "docker",
        "run",
        "--rm",
        "--privileged",  # DinD
        "-v",
        f"{repo_path.absolute()}:/repo",
        "-v",
        f"{repo_path.absolute()}:/testbed",
        tag,
    ]

    try:
        if output_file:
            with open(output_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    check=False,  # Don't raise exception on non-zero exit
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        else:
            result = subprocess.run(
                cmd,
                check=False,  # Don't raise exception on non-zero exit
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            logging.debug(f"Docker run output: {result.stdout}")

        return result.returncode
    except Exception as e:
        logging.error(f"Docker run failed: {e}")
        return -1


def apply_patch(repo_path: Path, patch_path: Path) -> bool:
    """
    Apply a patch to the repository.

    Args:
        repo_path: Path to the repository
        patch_path: Path to the patch file

    Returns:
        True if the patch was applied successfully, False otherwise
    """
    logging.info(f"Applying patch {patch_path} to repository at {repo_path}")
    try:
        result = subprocess.run(
            ["git", "apply", str(patch_path.absolute())],
            cwd=str(repo_path),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        logging.debug(f"Git apply output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to apply patch: {e.stderr}")
        return False


def get_pr_info(owner: str, repo: str, pr_number: int) -> Tuple[str, str]:
    """
    Get the base and head SHAs for a PR.

    Args:
        owner: Repository owner
        repo: Repository name
        pr_number: PR number

    Returns:
        Tuple of (base_sha, head_sha)
    """
    logging.info(f"Fetching PR information for {owner}/{repo}#{pr_number}")
    api_base = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    pr_info = github_get(api_base)

    base_sha = pr_info["base"]["sha"]
    head_sha = pr_info["head"]["sha"]
    logging.debug(f"PR base SHA: {base_sha}, head SHA: {head_sha}")

    return base_sha, head_sha


def get_commit_parent(owner: str, repo: str, sha: str) -> str:
    """
    Get the parent commit SHA for a commit.

    Args:
        owner: Repository owner
        repo: Repository name
        sha: Commit SHA

    Returns:
        Parent commit SHA
    """
    logging.info(f"Fetching commit information for {owner}/{repo} commit {sha}")
    info = github_get(f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}")

    parents = info.get("parents", [])
    if not parents:
        logging.error(f"Commit {sha} has no parents")
        raise ValueError("Root commit provided; cannot determine parent.")

    parent_sha = parents[0]["sha"]
    logging.debug(f"Parent commit SHA: {parent_sha}")

    return parent_sha


def process_url(
    url: str,
    patch_dir: Path,
    work_dir: Path,
    output_dir: Path,
    verify: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single PR or commit URL.

    Args:
        url: GitHub PR or commit URL
        patch_dir: Directory containing .patch files
        work_dir: Directory to clone repositories into
        output_dir: Directory to save output files
        verify: Whether to verify the base commit and pre-patch tests

    Returns:
        Tuple of (success, message)
    """
    logging.info(f"Processing URL: {url}")

    pr_match = PR_URL_RE.fullmatch(url.strip())
    commit_match = COMMIT_URL_RE.fullmatch(url.strip())
    if not (pr_match or commit_match):
        logging.error(f"URL format not recognized: {url}")
        raise ValueError(f"Unsupported URL format: {url}")

    owner = pr_match["owner"] if pr_match else commit_match["owner"]
    repo = pr_match["repo"] if pr_match else commit_match["repo"]
    repo_full = f"{owner}/{repo}"
    repo_url = f"https://github.com/{repo_full}.git"
    logging.info(f"Identified repository: {repo_full}")

    # Determine the base SHA and identifier
    if pr_match:
        pr_number = pr_match["number"]
        logging.info(f"Processing as PR #{pr_number}")
        base_sha, head_sha = get_pr_info(owner, repo, int(pr_number))
        identifier = f"{owner}_{repo}_{pr_number}"
    else:
        commit_sha = commit_match["sha"]
        logging.info(f"Processing as commit {commit_sha}")
        base_sha = get_commit_parent(owner, repo, commit_sha)
        identifier = f"{owner}_{repo}_{commit_sha}"

    # Set up paths
    test_eval_dir = Path("test_eval") / identifier
    test_eval_dir.mkdir(parents=True, exist_ok=True)

    # Find the corresponding patch file
    patch_file = next((p for p in patch_dir.glob(f"{identifier}.patch")), None)
    if not patch_file:
        logging.error(f"No patch file found for {identifier}")
        return False, f"❌ {url}: No patch file found for {identifier}"

    # Check for check.sh or Dockerfile
    check_script_path = test_eval_dir / "check.sh"
    dockerfile_path = test_eval_dir / "Dockerfile"

    if not (check_script_path.exists() or dockerfile_path.exists()):
        logging.error(f"Neither check.sh nor Dockerfile found in {test_eval_dir}")
        return (
            False,
            f"❌ {url}: Neither check.sh nor Dockerfile found in {test_eval_dir}",
        )

    # Determine if we're using check.sh or Docker
    using_check_script = check_script_path.exists()

    # Check for test diff
    test_diff_path = test_eval_dir / "test.diff"
    if not using_check_script and not test_diff_path.exists():
        logging.error(f"No test diff found at {test_diff_path}")
        return False, f"❌ {url}: No test diff found at {test_diff_path}"

    # Clone repository
    local_repo = work_dir / identifier
    if not local_repo.exists():
        logging.info(f"Cloning repository {repo_url} to {local_repo}")
        repo = Repo.clone_from(repo_url, local_repo)
    else:
        logging.info(f"Repository already cloned at {local_repo}")
        repo = Repo(local_repo)
        repo.git.reset("--hard", base_sha)
        repo.git.clean("-fdx")

    # Checkout base commit
    logging.info(f"Checking out base SHA {base_sha}")
    repo.git.checkout(base_sha)

    # If using Docker, build the image
    docker_tag = None
    if not using_check_script:
        docker_tag = f"test-eval-{identifier}".lower()
        if not build_docker_image(dockerfile_path, docker_tag):
            return False, f"❌ {url}: Failed to build Docker image"

    if verify:
        # Run tests with base commit (should succeed)
        if not using_check_script:
            base_exit_code = run_docker_container(docker_tag, local_repo)

            if base_exit_code != 0:
                logging.error(f"Base test failed with exit code {base_exit_code}")
                return (
                    False,
                    f"❌ {url}: Base test failed with exit code {base_exit_code}",
                )

        # Run tests with test diff applied (should fail)
        if using_check_script:
            test_exit_code = run_check_script(check_script_path, local_repo)
        else:
            # Apply test diff (should make tests fail)
            if not apply_patch(local_repo, test_diff_path):
                return False, f"❌ {url}: Failed to apply test diff"
            test_exit_code = run_docker_container(docker_tag, local_repo)

        if test_exit_code == 0:
            logging.error("Test diff did not cause tests to fail")
            return False, f"❌ {url}: Test diff did not cause tests to fail"

        # Reset to base commit before applying the fixing patch
        logging.info(f"Resetting to base SHA {base_sha}")
        repo.git.checkout(base_sha, force=True)
        repo.git.clean("-fdx")

    if not using_check_script:
        # Apply test diff
        if not apply_patch(local_repo, test_diff_path):
            return False, f"❌ {url}: Failed to apply test diff"

    # Apply fixing patch
    if not apply_patch(local_repo, patch_file):
        return False, f"❌ {url}: Failed to apply fixing patch"

    # Run tests with fixing patch applied (should succeed)
    output_file = output_dir / f"{identifier}_output.txt"
    if using_check_script:
        fix_exit_code = run_check_script(check_script_path, local_repo, output_file)
    else:
        fix_exit_code = run_docker_container(docker_tag, local_repo, output_file)

    if fix_exit_code == 0:
        logging.info(f"Success! Fixing patch resolved the issue.")
        return True, f"✅ {url}: Fixing patch successfully resolved the issue"
    else:
        logging.error(
            f"Fixing patch did not resolve the issue, exit code: {fix_exit_code}"
        )
        return (
            False,
            f"❌ {url}: Fixing patch did not resolve the issue, exit code: {fix_exit_code}",
        )


def read_urls(path: Path) -> Iterable[str]:
    """
    Read URLs from a file, skipping empty lines and comments.

    Args:
        path: Path to the file containing URLs

    Returns:
        Iterable of URLs
    """
    logging.info(f"Reading URLs from {path}")
    with path.open() as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line.strip()


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Test evaluation for GitHub PRs and commits."
    )
    parser.add_argument("url_file", type=Path, help="File with PR/commit URLs.")
    parser.add_argument(
        "patch_dir", type=Path, help="Directory containing .patch files."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_results"),
        help="Directory to store test outputs (default: test_results)",
    )
    parser.add_argument("--tmpdir", type=Path, help="Directory to store clones.")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify base commit and test patch behavior before applying fixing patch",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(message)s",
    )

    urls = list(read_urls(args.url_file))
    if not urls:
        logging.error("No URLs to process.")
        sys.exit(1)

    if args.tmpdir:
        work_dir = args.tmpdir
        logging.info(f"Using specified temporary directory: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory(prefix="test-eval-")
        work_dir = Path(tmp.name)
        logging.info(f"Created temporary directory: {work_dir}")
        cleanup = True

    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving test outputs to {output_dir}")

    successes = 0
    for i, u in enumerate(tqdm(urls, desc="Processing")):
        logging.info(f"Processing URL {i+1}/{len(urls)}: {u}")
        try:
            ok, msg = process_url(u, args.patch_dir, work_dir, output_dir, args.verify)
            print(msg)
            if ok:
                successes += 1
                logging.info(f"Successfully processed {u}")
            else:
                logging.info(f"Failed to process {u}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception(f"Error processing {u}")
            print(f"❌ {u}: Error: {str(e)}")

    print("\n==== SUMMARY ====")
    print(f"Processed : {len(urls)}")
    print(f"Successful: {successes}")
    logging.info(f"Final results: processed {len(urls)} URLs, {successes} successful")

    if cleanup:
        logging.debug(f"Cleaning up temporary directory: {work_dir}")
        tmp.cleanup()


if __name__ == "__main__":
    main()
