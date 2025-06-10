#!/usr/bin/env python3
"""
Script to automatically create GitHub PRs in SM-100-Bench repositories.

This script:
1. Loads a CSV of GitHub PR/commit URLs and commit SHAs
2. For each entry, creates a PR in the corresponding SM-100-Bench repository
3. Uses the 'pre' branch as the base (created by gh_pr_branch_sync.py)
4. Creates a new branch with the specified commit changes
5. Uses the commit message as the PR description
"""

import argparse
import csv
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Tuple, Optional

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


def get_commit_info(owner: str, repo: str, sha: str) -> dict:
    """
    Get commit information from GitHub API.

    Args:
        owner: Repository owner
        repo: Repository name
        sha: Commit SHA

    Returns:
        Commit information dictionary
    """
    logging.info(f"Fetching commit information for {owner}/{repo} commit {sha}")
    api_url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"

    response = requests.get(api_url, headers=HEADERS)
    if response.status_code != 200:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")

    return response.json()


def check_pr_exists(repo_identifier: str, head_branch: str) -> Optional[dict]:
    """
    Check if a PR already exists with the given head branch.

    Args:
        repo_identifier: Repository identifier
        head_branch: Head branch name

    Returns:
        PR information if exists, None otherwise
    """
    logging.info(
        f"Checking for existing PR with head branch '{head_branch}' in {SM100_ORG}/{repo_identifier}"
    )

    api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/pulls"
    params = {"state": "open", "head": f"{SM100_ORG}:{head_branch}", "base": "pre"}

    try:
        response = requests.get(api_url, headers=HEADERS, params=params)
        if response.status_code == 200:
            prs = response.json()
            if prs:
                logging.info(f"Found existing PR: {prs[0]['html_url']}")
                return prs[0]
        return None
    except Exception as e:
        logging.warning(f"Error checking for existing PR: {e}")
        return None


def create_github_pr(
    repo_identifier: str, head_branch: str, title: str, body: str
) -> dict:
    """
    Create a GitHub PR using the API.

    Args:
        repo_identifier: Repository identifier
        head_branch: Head branch name
        title: PR title
        body: PR body/description

    Returns:
        Created PR information
    """
    logging.info(f"Creating PR in {SM100_ORG}/{repo_identifier}")

    api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/pulls"

    pr_data = {"title": title, "body": body, "head": head_branch, "base": "pre"}

    response = requests.post(api_url, headers=HEADERS, json=pr_data)
    if response.status_code != 201:
        raise RuntimeError(f"GitHub API error {response.status_code}: {response.text}")

    return response.json()


def process_entry(
    url: str,
    commit_sha: str,
    work_dir: Path,
    suffix: str = "",
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

        # Parse URL to get original owner/repo
        pr_match = PR_URL_RE.fullmatch(url.strip())
        commit_match = COMMIT_URL_RE.fullmatch(url.strip())

        owner = pr_match["owner"] if pr_match else commit_match["owner"]
        repo = pr_match["repo"] if pr_match else commit_match["repo"]

        # Get commit information for the title and description
        commit_info = get_commit_info(owner, repo, commit_sha)
        commit_message = commit_info["commit"]["message"]

        # Use first line as title, rest as body
        lines = commit_message.strip().split("\n")
        pr_title = lines[0]
        if suffix:
            pr_title = f"{pr_title} [{suffix}]"
        pr_body = "\n".join(lines[1:]).strip() if len(lines) > 1 else commit_message

        # Create a branch name based on the commit SHA
        head_branch = f"auto-pr-{commit_sha[:8]}"
        if suffix:
            head_branch = f"{head_branch}-{suffix}"

        # Check if PR already exists
        existing_pr = check_pr_exists(repo_identifier, head_branch)
        if existing_pr:
            return True, f"PR already exists: {existing_pr['html_url']}"

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

        # Check if 'pre' branch exists
        try:
            repo.git.checkout("pre")
            logging.info("Checked out 'pre' branch")
        except Exception as e:
            logging.error(f"Failed to checkout 'pre' branch: {e}")
            return (
                False,
                f"Failed to checkout 'pre' branch. Make sure gh_pr_branch_sync.py has been run first: {str(e)}",
            )

        # Create new branch for the PR
        logging.info(f"Creating branch '{head_branch}'")
        try:
            repo.git.checkout("-b", head_branch)
        except Exception as e:
            logging.error(f"Failed to create branch '{head_branch}': {e}")
            return False, f"Failed to create branch '{head_branch}': {str(e)}"

        # Cherry-pick the commit
        logging.info(f"Cherry-picking commit {commit_sha}")
        try:
            repo.git.cherry_pick(commit_sha)
        except Exception as e:
            logging.error(f"Failed to cherry-pick commit: {e}")
            return False, f"Failed to cherry-pick commit {commit_sha}: {str(e)}"

        # Set up authenticated remote URL for pushing
        auth_url = sm100_repo_url.replace(
            "https://", f"https://{SM100_GITHUB_SYNC_PAT}@"
        )
        repo.git.remote("set-url", "origin", auth_url)

        # Push the branch
        logging.info(f"Pushing branch '{head_branch}' to remote")
        try:
            repo.git.push("origin", head_branch)
        except Exception as e:
            logging.error(f"Failed to push branch '{head_branch}': {e}")
            return False, f"Failed to push branch '{head_branch}': {str(e)}"

        # Create the PR
        logging.info("Creating GitHub PR")
        try:
            pr_info = create_github_pr(repo_identifier, head_branch, pr_title, pr_body)
            pr_url = pr_info["html_url"]
            logging.info(f"Successfully created PR: {pr_url}")

            # Clean up the repository clone immediately after successful PR creation to save disk space
            if local_repo_path.exists():
                logging.debug(f"Cleaning up repository clone at {local_repo_path}")
                shutil.rmtree(local_repo_path)

            return True, f"Successfully created PR: {pr_url}"
        except Exception as e:
            logging.error(f"Failed to create PR: {e}")
            return False, f"Failed to create PR: {str(e)}"

    except Exception as e:
        logging.error(f"Error processing {url} with SHA {commit_sha}: {e}")
        return False, f"Error processing {url} with SHA {commit_sha}: {str(e)}"


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Automatically create GitHub PRs in SM-100-Bench repositories"
    )
    parser.add_argument(
        "csv_file", type=Path, help="CSV file with PR/commit URLs and commit SHAs."
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="",
        help="Suffix to append to branch names and PR titles for independent evaluation runs.",
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
        tmp = tempfile.TemporaryDirectory(prefix="gh-auto-pr-")
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

            ok, msg = process_entry(url, commit_sha, entry_work_dir, args.suffix)

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
        f"Final results: processed {len(entries)} entries, successfully created {successes} PRs"
    )

    if cleanup:
        logging.debug(f"Cleaning up temporary directory: {work_dir}")
        tmp.cleanup()


if __name__ == "__main__":
    main()
