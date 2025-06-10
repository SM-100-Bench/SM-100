#!/usr/bin/env python3
"""
Script to create 'pre' branches in SM-100-Bench repositories.

This script:
1. Loads a CSV of GitHub PR/commit URLs and commit SHAs
2. Clones the corresponding repo from the SM-100-Bench org on GitHub
3. Creates a branch called "pre" which points to {commit_sha}~ (parent of commit_sha)
4. Pushes the branch to the remote repository
"""

import argparse
import csv
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple

import requests
from git import Repo
from tqdm import tqdm

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
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


def read_csv_entries(path: Path) -> Iterable[Tuple[str, str]]:
    """
    Read CSV file with columns: pr/commit URL, commit sha
    Returns tuples of (url, commit_sha)
    """
    logging.info(f"Reading CSV entries from {path}")
    with path.open() as f:
        reader = csv.reader(f)
        # Skip header row if it exists
        first_row = next(reader, None)
        if first_row and (
            first_row[0].lower().startswith("url")
            or first_row[0].lower().startswith("pr")
            or first_row[0].lower().startswith("commit")
        ):
            # This looks like a header row, skip it
            pass
        else:
            # This is data, yield it
            if first_row and len(first_row) >= 2:
                yield first_row[0].strip(), first_row[1].strip()

        # Process remaining rows
        for row in reader:
            if len(row) >= 2 and row[0].strip() and not row[0].strip().startswith("#"):
                yield row[0].strip(), row[1].strip()


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


def get_repo_identifier(url: str) -> str:
    """
    Get the repository identifier from a PR or commit URL.

    Args:
        url: GitHub PR or commit URL

    Returns:
        Repository identifier in format: owner_repo_identifier
    """
    pr_match = PR_URL_RE.fullmatch(url.strip())
    commit_match = COMMIT_URL_RE.fullmatch(url.strip())

    if not (pr_match or commit_match):
        raise ValueError(f"URL format not recognized: {url}")

    owner = pr_match["owner"] if pr_match else commit_match["owner"]
    repo = pr_match["repo"] if pr_match else commit_match["repo"]

    if pr_match:
        pr_number = pr_match["number"]
        identifier = f"{owner}_{repo}_{pr_number}"
    else:
        commit_sha = commit_match["sha"]
        identifier = f"{owner}_{repo}_{commit_sha}"

    return identifier


def check_branch_exists(repo_identifier: str, branch_name: str) -> bool:
    """
    Check if a branch exists in the SM-100-Bench repository.

    Args:
        repo_identifier: Repository identifier
        branch_name: Branch name to check

    Returns:
        True if branch exists, False otherwise
    """
    logging.info(
        f"Checking if branch '{branch_name}' exists in {SM100_ORG}/{repo_identifier}"
    )

    api_url = (
        f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/branches/{branch_name}"
    )
    response = requests.get(api_url, headers=HEADERS)

    if response.status_code == 200:
        logging.debug(f"Branch '{branch_name}' exists in {SM100_ORG}/{repo_identifier}")
        return True
    elif response.status_code == 404:
        logging.debug(
            f"Branch '{branch_name}' does not exist in {SM100_ORG}/{repo_identifier}"
        )
        return False
    else:
        logging.warning(f"GitHub API error {response.status_code}: {response.text}")
        return False


def process_entry(
    url: str,
    commit_sha: str,
    work_dir: Path,
) -> Tuple[bool, str]:
    """
    Process a single PR or commit URL with specified commit SHA.

    Args:
        url: GitHub PR or commit URL
        commit_sha: Commit SHA from CSV
        work_dir: Directory to clone repositories into

    Returns:
        Tuple of (success, message)
    """
    logging.info(f"Processing URL: {url} with commit SHA: {commit_sha}")

    try:
        # Get repository identifier
        repo_identifier = get_repo_identifier(url)
        logging.info(f"Repository identifier: {repo_identifier}")

        # Check if 'pre' branch already exists
        if check_branch_exists(repo_identifier, "pre"):
            logging.info(
                f"Branch 'pre' already exists in {SM100_ORG}/{repo_identifier}"
            )
            return True, f"Branch 'pre' already exists in {SM100_ORG}/{repo_identifier}"

        # Parse URL to get original owner/repo
        pr_match = PR_URL_RE.fullmatch(url.strip())
        commit_match = COMMIT_URL_RE.fullmatch(url.strip())

        owner = pr_match["owner"] if pr_match else commit_match["owner"]
        repo = pr_match["repo"] if pr_match else commit_match["repo"]

        # Get parent commit SHA (commit_sha~)
        parent_sha = get_commit_parent(owner, repo, commit_sha)
        logging.info(f"Parent commit SHA: {parent_sha}")

        # Clone the SM-100-Bench repository
        sm100_repo_url = f"https://github.com/{SM100_ORG}/{repo_identifier}.git"
        local_repo_path = work_dir / repo_identifier

        if local_repo_path.exists():
            logging.info(f"Removing existing repository at {local_repo_path}")
            shutil.rmtree(local_repo_path)

        logging.info(f"Cloning repository {sm100_repo_url} to {local_repo_path}")
        try:
            repo = Repo.clone_from(sm100_repo_url, local_repo_path)
        except Exception as e:
            logging.error(f"Failed to clone repository: {e}")
            return (
                False,
                f"Failed to clone repository {SM100_ORG}/{repo_identifier}: {str(e)}",
            )

        # Configure Git credentials
        repo.git.config("user.name", "SM-100-Bench Bot")
        repo.git.config("user.email", "noreply@example.com")

        # Checkout the parent commit
        logging.info(f"Checking out parent commit SHA {parent_sha}")
        try:
            repo.git.checkout(parent_sha)
        except Exception as e:
            logging.error(f"Failed to checkout parent SHA: {e}")
            return False, f"Failed to checkout parent SHA {parent_sha}: {str(e)}"

        # Create 'pre' branch
        logging.info("Creating 'pre' branch")
        try:
            repo.git.checkout("-b", "pre")
        except Exception as e:
            logging.error(f"Failed to create 'pre' branch: {e}")
            return False, f"Failed to create 'pre' branch: {str(e)}"

        # Set up authenticated remote URL for pushing
        auth_url = sm100_repo_url.replace(
            "https://", f"https://{SM100_GITHUB_SYNC_PAT}@"
        )
        repo.git.remote("set-url", "origin", auth_url)

        # Push the 'pre' branch
        logging.info("Pushing 'pre' branch to remote")
        try:
            repo.git.push("origin", "pre")
        except Exception as e:
            logging.error(f"Failed to push 'pre' branch: {e}")
            return False, f"Failed to push 'pre' branch: {str(e)}"

        logging.info(
            f"Successfully created and pushed 'pre' branch for {repo_identifier}"
        )
        return (
            True,
            f"Successfully created 'pre' branch for {repo_identifier} pointing to {parent_sha[:7]}",
        )

    except Exception as e:
        logging.error(f"Error processing {url} with SHA {commit_sha}: {e}")
        return False, f"Error processing {url} with SHA {commit_sha}: {str(e)}"


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Create 'pre' branches in SM-100-Bench repositories pointing to commit_sha~"
    )
    parser.add_argument(
        "csv_file", type=Path, help="CSV file with PR/commit URLs and commit SHAs."
    )
    parser.add_argument("--tmpdir", type=Path, help="Directory to store clones.")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(message)s",
    )

    entries = list(read_csv_entries(args.csv_file))
    if not entries:
        logging.error("No entries to process.")
        sys.exit(1)

    if args.tmpdir:
        work_dir = args.tmpdir
        logging.info(f"Using specified temporary directory: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory(prefix="gh-pr-branch-sync-")
        work_dir = Path(tmp.name)
        logging.info(f"Created temporary directory: {work_dir}")
        cleanup = True

    successes = 0
    for i, (url, commit_sha) in enumerate(tqdm(entries, desc="Processing")):
        logging.info(
            f"Processing entry {i+1}/{len(entries)}: {url} with SHA {commit_sha}"
        )
        try:
            # Create a unique temporary directory for this entry
            entry_work_dir = work_dir / f"entry_{i}"
            entry_work_dir.mkdir(parents=True, exist_ok=True)

            ok, msg = process_entry(url, commit_sha, entry_work_dir)

            # Clean up the temporary directory
            if entry_work_dir.exists():
                shutil.rmtree(entry_work_dir)

            if ok:
                successes += 1
                print(f"✅ {msg}")
                logging.info(f"Successfully processed {url} with SHA {commit_sha}")
            else:
                print(f"❌ {msg}")
                logging.info(f"Failed to process {url} with SHA {commit_sha}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception("Error processing %s with SHA %s", url, commit_sha)
            print(f"❌ Error processing {url} with SHA {commit_sha}: {str(e)}")

    print("\n==== SUMMARY ====")
    print(f"Processed : {len(entries)}")
    print(f"Successful: {successes}")
    logging.info(
        f"Final results: processed {len(entries)} entries, successfully processed {successes}"
    )

    if cleanup:
        logging.debug(f"Cleaning up temporary directory: {work_dir}")
        tmp.cleanup()


if __name__ == "__main__":
    main()
