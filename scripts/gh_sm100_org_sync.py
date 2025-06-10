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
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from git import Repo
from tqdm import tqdm

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    github_get,
)

# GitHub API base URL
GITHUB_API_BASE = "https://api.github.com"

# Organization name for synced repositories
SM100_ORG = "SM-100-Bench"

# Environment variable for GitHub Personal Access Token
SM100_GITHUB_SYNC_PAT = os.getenv("SM100_GITHUB_SYNC_PAT")
if not SM100_GITHUB_SYNC_PAT:
    logging.error("Environment variable SM100_GITHUB_SYNC_PAT is not set")
    sys.exit(1)

# Headers for GitHub API requests
HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {SM100_GITHUB_SYNC_PAT}",
    "X-GitHub-Api-Version": "2022-11-28",
}


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
    api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/pulls/{pr_number}"

    response = requests.get(api_url, headers=HEADERS)
    if response.status_code != 200:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")

    pr_info = response.json()
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
    api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"

    response = requests.get(api_url, headers=HEADERS)
    if response.status_code != 200:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")

    commit_info = response.json()
    parents = commit_info.get("parents", [])
    if not parents:
        logging.error(f"Commit {sha} has no parents")
        raise ValueError("Root commit provided; cannot determine parent.")

    parent_sha = parents[0]["sha"]
    logging.debug(f"Parent commit SHA: {parent_sha}")

    return parent_sha


def get_branch_commit_sha(
    org: str, repo_name: str, branch: str = "main"
) -> Optional[str]:
    """
    Get the commit SHA of a branch in a repository.

    Args:
        org: Organization name
        repo_name: Repository name
        branch: Branch name (default: main)

    Returns:
        The commit SHA if the branch exists, None otherwise
    """
    logging.info(f"Checking commit SHA of {branch} branch in {org}/{repo_name}")

    api_url = f"{GITHUB_API_BASE}/repos/{org}/{repo_name}/branches/{branch}"
    response = requests.get(api_url, headers=HEADERS)

    if response.status_code == 200:
        branch_info = response.json()
        commit_sha = branch_info["commit"]["sha"]
        logging.debug(f"Branch {branch} commit SHA: {commit_sha}")
        return commit_sha
    elif response.status_code == 404:
        logging.debug(f"Branch {branch} not found in {org}/{repo_name}")
        return None
    else:
        logging.warning(f"GitHub API error {response.status_code}: {response.text}")
        return None


def create_or_update_repo(repo_name: str) -> str:
    """
    Create a repository in the SM-100-Bench organization if it doesn't exist,
    or update its settings if it does.

    Args:
        repo_name: Name of the repository to create or update

    Returns:
        The clone URL of the repository
    """
    logging.info(f"Checking if repository {SM100_ORG}/{repo_name} exists")

    # Check if repo exists
    check_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_name}"
    response = requests.get(check_url, headers=HEADERS)

    if response.status_code == 200:
        logging.info(f"Repository {SM100_ORG}/{repo_name} already exists")
        repo_info = response.json()

        # Update repository settings
        update_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_name}"
        update_data = {
            "has_issues": False,
            "has_wiki": False,
            "has_projects": False,
            "has_downloads": True,
            "allow_squash_merge": True,
            "allow_merge_commit": True,
            "allow_rebase_merge": True,
        }

        update_response = requests.patch(update_url, headers=HEADERS, json=update_data)
        if update_response.status_code != 200:
            logging.warning(
                f"Failed to update repository settings: {update_response.text}"
            )

        # Disable GitHub Actions if needed
        actions_url = (
            f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_name}/actions/permissions"
        )
        actions_data = {"enabled": False}

        actions_response = requests.put(actions_url, headers=HEADERS, json=actions_data)
        if actions_response.status_code != 204:
            logging.warning(
                f"Failed to disable GitHub Actions: {actions_response.text}"
            )

        return repo_info["clone_url"]

    elif response.status_code == 404:
        logging.info(f"Creating repository {SM100_ORG}/{repo_name}")

        # Create repository
        create_url = f"{GITHUB_API_BASE}/orgs/{SM100_ORG}/repos"
        create_data = {
            "name": repo_name,
            "private": False,
            "has_issues": False,
            "has_wiki": False,
            "has_projects": False,
            "has_downloads": True,
            "auto_init": False,
            "allow_squash_merge": True,
            "allow_merge_commit": True,
            "allow_rebase_merge": True,
        }

        create_response = requests.post(create_url, headers=HEADERS, json=create_data)
        if create_response.status_code != 201:
            raise RuntimeError(f"Failed to create repository: {create_response.text}")

        repo_info = create_response.json()

        # Disable GitHub Actions
        actions_url = (
            f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_name}/actions/permissions"
        )
        actions_data = {"enabled": False}

        actions_response = requests.put(actions_url, headers=HEADERS, json=actions_data)
        if actions_response.status_code != 204:
            logging.warning(
                f"Failed to disable GitHub Actions: {actions_response.text}"
            )

        return repo_info["clone_url"]

    else:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")


def process_url(url: str, work_dir: Path) -> bool:
    """
    Process a single PR or commit URL.

    Args:
        url: GitHub PR or commit URL
        work_dir: Directory to clone repositories into

    Returns:
        True if successful, False otherwise
    """
    logging.info(f"Processing URL: {url}")

    pr_match = PR_URL_RE.fullmatch(url.strip())
    commit_match = COMMIT_URL_RE.fullmatch(url.strip())
    if not (pr_match or commit_match):
        logging.error(f"URL format not recognized: {url}")
        return False

    owner = pr_match["owner"] if pr_match else commit_match["owner"]
    repo = pr_match["repo"] if pr_match else commit_match["repo"]
    repo_full = f"{owner}/{repo}"
    repo_url = f"https://github.com/{repo_full}.git"
    logging.info(f"Identified repository: {repo_full}")

    # Determine the base SHA and identifier
    if pr_match:
        pr_number = pr_match["number"]
        logging.info(f"Processing as PR #{pr_number}")
        base_sha, _ = get_pr_info(owner, repo, int(pr_number))
        identifier = f"{owner}_{repo}_{pr_number}"
    else:
        commit_sha = commit_match["sha"]
        logging.info(f"Processing as commit {commit_sha}")
        base_sha = get_commit_parent(owner, repo, commit_sha)
        identifier = f"{owner}_{repo}_{commit_sha}"

    # Create or update repository in SM-100-Bench organization
    try:
        target_repo_url = create_or_update_repo(identifier)
        logging.info(f"Target repository URL: {target_repo_url}")
    except Exception as e:
        logging.error(f"Failed to create or update repository: {e}")
        return False

    # Check if the target repository already has the expected commit
    target_commit_sha = get_branch_commit_sha(SM100_ORG, identifier)
    if target_commit_sha:
        # Compare with the expected base SHA
        if target_commit_sha == base_sha:
            logging.info(
                f"Repository {SM100_ORG}/{identifier} already has the expected commit {base_sha}"
            )
            print(f"✓ {url}: Repository already up to date with commit {base_sha[:7]}")
            return True
        else:
            logging.info(
                f"Repository {SM100_ORG}/{identifier} has commit {target_commit_sha}, "
                f"but expected {base_sha}"
            )

    # Clone source repository
    local_repo_path = work_dir / identifier
    if local_repo_path.exists():
        logging.info(f"Removing existing repository at {local_repo_path}")
        shutil.rmtree(local_repo_path)

    logging.info(f"Cloning repository {repo_url} to {local_repo_path}")
    try:
        source_repo = Repo.clone_from(repo_url, local_repo_path)
    except Exception as e:
        logging.error(f"Failed to clone repository: {e}")
        return False

    # Checkout base commit
    logging.info(f"Checking out base SHA {base_sha}")
    try:
        source_repo.git.checkout(base_sha)
    except Exception as e:
        logging.error(f"Failed to checkout base SHA: {e}")
        return False

    # Set up target repository
    logging.info(f"Setting up target repository")
    try:
        # Configure Git credentials for push
        source_repo.git.config("user.name", "SM-100-Bench Bot")
        source_repo.git.config("user.email", "noreply@example.com")

        # Create and checkout a new branch (will be renamed to main)
        source_repo.git.checkout("-b", "temp-main")

        # Add target repository as remote
        auth_url = target_repo_url.replace(
            "https://", f"https://{SM100_GITHUB_SYNC_PAT}@"
        )
        source_repo.git.remote("add", "target", auth_url)

        # Force push to target repository
        source_repo.git.push("--force", "target", "temp-main:main")

        # Delete all remote branches except main
        remote_branches = source_repo.git.ls_remote("--heads", "target").splitlines()
        for branch in remote_branches:
            if "refs/heads/main" not in branch:
                branch_name = branch.split("refs/heads/")[1]
                source_repo.git.push("target", f":refs/heads/{branch_name}")

        # Delete all remote tags
        remote_tags = source_repo.git.ls_remote("--tags", "target").splitlines()
        for tag in remote_tags:
            tag_name = tag.split("refs/tags/")[1]
            source_repo.git.push("target", f":refs/tags/{tag_name}")

        logging.info(f"Successfully synced repository to {SM100_ORG}/{identifier}")
        return True

    except Exception as e:
        logging.error(f"Failed to set up target repository: {e}")
        return False


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Sync GitHub repositories to SM-100-Bench organization."
    )
    parser.add_argument("url_file", type=Path, help="File with PR/commit URLs.")
    parser.add_argument("--tmpdir", type=Path, help="Directory to store clones.")
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
        tmp = tempfile.TemporaryDirectory(prefix="sm100-sync-")
        work_dir = Path(tmp.name)
        logging.info(f"Created temporary directory: {work_dir}")
        cleanup = True

    successes = 0
    for i, url in enumerate(tqdm(urls, desc="Processing")):
        logging.info(f"Processing URL {i+1}/{len(urls)}: {url}")
        try:
            if process_url(url, work_dir):
                successes += 1
                print(f"✅ {url}: Successfully synced to SM-100-Bench organization")
            else:
                print(f"❌ {url}: Failed to sync to SM-100-Bench organization")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception(f"Error processing {url}")
            print(f"❌ {url}: Error: {str(e)}")

    print("\n==== SUMMARY ====")
    print(f"Processed : {len(urls)}")
    print(f"Successful: {successes}")
    logging.info(f"Final results: processed {len(urls)} URLs, {successes} successful")

    if cleanup:
        logging.debug(f"Cleaning up temporary directory: {work_dir}")
        tmp.cleanup()


if __name__ == "__main__":
    main()
