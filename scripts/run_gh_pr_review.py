#!/usr/bin/env python3
"""
Script to trigger PR reviews and collect review comments.

This script:
1. Loads a CSV of GitHub PR/commit URLs and commit SHAs
2. For each entry, finds the corresponding PR created by gh_auto_pr.py
3. Comments a trigger phrase on the PR if not already commented
4. Waits for reply comments and/or review comments to be posted
5. Collects all review comments and saves them to a JSON file
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
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


def read_csv_entries(path: Path) -> List[Tuple[str, str]]:
    """
    Read CSV file with columns: pr/commit URL, commit sha
    Returns list of tuples (url, commit_sha)
    """
    logging.info(f"Reading CSV entries from {path}")
    entries = []

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
            # This is data, add it
            if first_row and len(first_row) >= 2:
                entries.append((first_row[0].strip(), first_row[1].strip()))

        # Process remaining rows
        for row in reader:
            if len(row) >= 2 and row[0].strip() and not row[0].strip().startswith("#"):
                entries.append((row[0].strip(), row[1].strip()))

    return entries


def get_repo_identifier(url: str) -> str:
    """
    Get the repository identifier from a PR or commit URL.
    This should match the logic in gh_auto_pr.py
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


def find_auto_pr(
    repo_identifier: str, commit_sha: str, suffix: str = ""
) -> Optional[Dict]:
    """
    Find the PR created by gh_auto_pr.py for the given commit.

    Args:
        repo_identifier: Repository identifier
        commit_sha: Commit SHA
        suffix: Branch suffix used in gh_auto_pr.py

    Returns:
        PR information if found, None otherwise
    """
    # Construct the expected branch name (matches gh_auto_pr.py logic)
    head_branch = f"auto-pr-{commit_sha[:8]}"
    if suffix:
        head_branch = f"{head_branch}-{suffix}"

    logging.info(
        f"Looking for PR with head branch '{head_branch}' in {SM100_ORG}/{repo_identifier}"
    )

    api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/pulls"
    params = {"state": "open", "head": f"{SM100_ORG}:{head_branch}", "base": "pre"}

    try:
        response = requests.get(api_url, headers=HEADERS, params=params)
        if response.status_code == 200:
            prs = response.json()
            if prs:
                logging.info(f"Found auto-created PR: {prs[0]['html_url']}")
                return prs[0]

        # Also check closed PRs in case it was merged/closed
        params["state"] = "closed"
        response = requests.get(api_url, headers=HEADERS, params=params)
        if response.status_code == 200:
            prs = response.json()
            if prs:
                logging.info(f"Found auto-created PR (closed): {prs[0]['html_url']}")
                return prs[0]

        return None
    except Exception as e:
        logging.warning(f"Error finding auto PR: {e}")
        return None


def check_trigger_comment_exists(
    repo_identifier: str, pr_number: int, trigger_phrase: str
) -> bool:
    """
    Check if the trigger phrase has already been commented on the PR.

    Args:
        repo_identifier: Repository identifier
        pr_number: PR number
        trigger_phrase: The trigger phrase to look for

    Returns:
        True if trigger comment exists, False otherwise
    """
    logging.info(f"Checking for existing trigger comment on PR #{pr_number}")

    api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/issues/{pr_number}/comments"

    try:
        response = requests.get(api_url, headers=HEADERS)
        if response.status_code == 200:
            comments = response.json()
            for comment in comments:
                if trigger_phrase.strip() in comment["body"]:
                    logging.info(
                        f"Found existing trigger comment: {comment['html_url']}"
                    )
                    return True
        return False
    except Exception as e:
        logging.warning(f"Error checking for trigger comment: {e}")
        return False


def delete_all_pr_comments(repo_identifier: str, pr_number: int) -> Tuple[int, int]:
    """
    Delete all comments on a PR (both issue comments and review comments).

    Args:
        repo_identifier: Repository identifier
        pr_number: PR number

    Returns:
        Tuple of (deleted_issue_comments, deleted_review_comments)
    """
    logging.info(f"Deleting all comments on PR #{pr_number}")

    deleted_issue_comments = 0
    deleted_review_comments = 0

    # Delete issue comments
    try:
        api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/issues/{pr_number}/comments"
        response = requests.get(api_url, headers=HEADERS)
        if response.status_code == 200:
            comments = response.json()
            for comment in comments:
                comment_id = comment["id"]
                delete_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/issues/comments/{comment_id}"
                delete_response = requests.delete(delete_url, headers=HEADERS)
                if delete_response.status_code == 204:
                    deleted_issue_comments += 1
                    logging.debug(f"Deleted issue comment {comment_id}")
                else:
                    logging.warning(
                        f"Failed to delete issue comment {comment_id}: {delete_response.status_code}"
                    )
    except Exception as e:
        logging.warning(f"Error deleting issue comments: {e}")

    # Delete review comments
    try:
        api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/pulls/{pr_number}/comments"
        response = requests.get(api_url, headers=HEADERS)
        if response.status_code == 200:
            comments = response.json()
            for comment in comments:
                comment_id = comment["id"]
                delete_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/pulls/comments/{comment_id}"
                delete_response = requests.delete(delete_url, headers=HEADERS)
                if delete_response.status_code == 204:
                    deleted_review_comments += 1
                    logging.debug(f"Deleted review comment {comment_id}")
                else:
                    logging.warning(
                        f"Failed to delete review comment {comment_id}: {delete_response.status_code}"
                    )
    except Exception as e:
        logging.warning(f"Error deleting review comments: {e}")

    logging.info(
        f"Deleted {deleted_issue_comments} issue comments and {deleted_review_comments} review comments from PR #{pr_number}"
    )
    return deleted_issue_comments, deleted_review_comments


def post_trigger_comment(
    repo_identifier: str, pr_number: int, trigger_phrase: str
) -> Optional[Dict]:
    """
    Post the trigger phrase as a comment on the PR.

    Args:
        repo_identifier: Repository identifier
        pr_number: PR number
        trigger_phrase: The trigger phrase to post

    Returns:
        Comment information if successful, None otherwise
    """
    logging.info(f"Posting trigger comment on PR #{pr_number}")

    api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/issues/{pr_number}/comments"

    comment_data = {"body": trigger_phrase}

    try:
        response = requests.post(api_url, headers=HEADERS, json=comment_data)
        if response.status_code == 201:
            comment_info = response.json()
            logging.info(
                f"Successfully posted trigger comment: {comment_info['html_url']}"
            )
            return comment_info
        else:
            logging.error(
                f"Failed to post comment: {response.status_code} {response.text}"
            )
            return None
    except Exception as e:
        logging.error(f"Error posting trigger comment: {e}")
        return None


def get_pr_comments_and_reviews(repo_identifier: str, pr_number: int) -> Dict:
    """
    Get all comments and reviews for a PR.

    Args:
        repo_identifier: Repository identifier
        pr_number: PR number

    Returns:
        Dictionary containing comments and reviews
    """
    logging.info(f"Fetching comments and reviews for PR #{pr_number}")

    result = {"issue_comments": [], "review_comments": [], "reviews": []}

    # Get issue comments
    try:
        api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/issues/{pr_number}/comments"
        response = requests.get(api_url, headers=HEADERS)
        if response.status_code == 200:
            result["issue_comments"] = response.json()
    except Exception as e:
        logging.warning(f"Error fetching issue comments: {e}")

    # Get review comments
    try:
        api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/pulls/{pr_number}/comments"
        response = requests.get(api_url, headers=HEADERS)
        if response.status_code == 200:
            result["review_comments"] = response.json()
    except Exception as e:
        logging.warning(f"Error fetching review comments: {e}")

    # Get reviews
    try:
        api_url = f"{GITHUB_API_BASE}/repos/{SM100_ORG}/{repo_identifier}/pulls/{pr_number}/reviews"
        response = requests.get(api_url, headers=HEADERS)
        if response.status_code == 200:
            result["reviews"] = response.json()
    except Exception as e:
        logging.warning(f"Error fetching reviews: {e}")

    return result


def wait_for_review_activity(
    repo_identifier: str,
    pr_number: int,
    trigger_comment_time: str,
    max_wait_minutes: int = 30,
    check_interval_seconds: int = 30,
) -> Dict:
    """
    Wait for new review activity after the trigger comment.

    Args:
        repo_identifier: Repository identifier
        pr_number: PR number
        trigger_comment_time: ISO timestamp of when trigger comment was posted
        max_wait_minutes: Maximum time to wait in minutes
        check_interval_seconds: How often to check for new activity

    Returns:
        Dictionary containing all comments and reviews
    """
    logging.info(
        f"Waiting for review activity on PR #{pr_number} (max {max_wait_minutes} minutes)"
    )

    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60

    while time.time() - start_time < max_wait_seconds:
        time.sleep(check_interval_seconds)

        current_data = get_pr_comments_and_reviews(repo_identifier, pr_number)

        # Check if there's new activity after the trigger comment
        new_activity = False
        for comment in current_data["issue_comments"]:
            if comment["created_at"] > trigger_comment_time:
                new_activity = True
                break

        for comment in current_data["review_comments"]:
            if comment["created_at"] > trigger_comment_time:
                new_activity = True
                break

        for review in current_data["reviews"]:
            if review["submitted_at"] and review["submitted_at"] > trigger_comment_time:
                new_activity = True
                break

        if new_activity:
            logging.info(f"Detected new review activity on PR #{pr_number}")
            return current_data

        elapsed_minutes = (time.time() - start_time) / 60
        logging.debug(
            f"Still waiting for review activity... ({elapsed_minutes:.1f}/{max_wait_minutes} minutes)"
        )

    logging.info(f"Timeout reached waiting for review activity on PR #{pr_number}")
    return get_pr_comments_and_reviews(repo_identifier, pr_number)


def save_review_results(
    data: Dict,
    repo_identifier: str,
    original_url: str,
    commit_sha: str,
    output_dir: Path,
) -> Path:
    """
    Save review results to a JSON file.

    Args:
        data: Review data to save
        repo_identifier: Repository identifier
        original_url: Original PR/commit URL
        commit_sha: Commit SHA
        output_dir: Output directory

    Returns:
        Path to saved file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename based on original URL (similar to existing results)
    safe_url = original_url.replace("https://", "").replace("/", "_").replace(":", "_")
    filename = f"pr_review_{safe_url}.json"
    output_path = output_dir / filename

    # Add metadata to the saved data
    result = {
        "metadata": {
            "original_url": original_url,
            "commit_sha": commit_sha,
            "repo_identifier": repo_identifier,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "review_data": data,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    logging.info(f"Saved review results to {output_path}")
    return output_path


def process_entry(
    url: str,
    commit_sha: str,
    suffix: str,
    trigger_phrase: str,
    output_dir: Path,
    max_wait_minutes: int = 30,
    clean: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single PR or commit URL entry.

    Args:
        url: GitHub PR or commit URL
        commit_sha: Commit SHA from CSV
        suffix: Branch suffix used in gh_auto_pr.py
        trigger_phrase: Phrase to comment to trigger review
        output_dir: Directory to save results
        max_wait_minutes: Maximum time to wait for review activity
        clean: Whether to delete all previous comments before posting trigger

    Returns:
        Tuple of (success, message)
    """
    logging.info(f"Processing URL: {url} with commit SHA: {commit_sha}")

    try:
        # Get repository identifier
        repo_identifier = get_repo_identifier(url)
        logging.info(f"Repository identifier: {repo_identifier}")

        # Find the auto-created PR
        auto_pr = find_auto_pr(repo_identifier, commit_sha, suffix)
        if not auto_pr:
            return (
                False,
                f"Could not find auto-created PR for {url} with SHA {commit_sha}",
            )

        pr_number = auto_pr["number"]
        pr_url = auto_pr["html_url"]

        # Delete all previous comments if clean flag is set
        if clean:
            deleted_issue, deleted_review = delete_all_pr_comments(
                repo_identifier, pr_number
            )
            logging.info(
                f"Cleaned PR #{pr_number}: deleted {deleted_issue} issue comments and {deleted_review} review comments"
            )

        # Check if trigger comment already exists (only if not cleaning)
        if not clean and check_trigger_comment_exists(
            repo_identifier, pr_number, trigger_phrase
        ):
            logging.info("Trigger comment already exists, skipping comment posting")
            trigger_comment_time = auto_pr[
                "created_at"
            ]  # Use PR creation time as fallback
        else:
            # Post trigger comment
            trigger_comment = post_trigger_comment(
                repo_identifier, pr_number, trigger_phrase
            )
            if not trigger_comment:
                return False, f"Failed to post trigger comment on {pr_url}"
            trigger_comment_time = trigger_comment["created_at"]

        # Wait for review activity
        review_data = wait_for_review_activity(
            repo_identifier, pr_number, trigger_comment_time, max_wait_minutes
        )

        # Save results
        output_path = save_review_results(
            review_data, repo_identifier, url, commit_sha, output_dir
        )

        return True, f"Successfully processed {pr_url}, saved results to {output_path}"

    except Exception as e:
        logging.error(f"Error processing {url} with SHA {commit_sha}: {e}")
        return False, f"Error processing {url} with SHA {commit_sha}: {str(e)}"


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Trigger PR reviews and collect review comments"
    )
    parser.add_argument(
        "csv_file", type=Path, help="CSV file with PR/commit URLs and commit SHAs"
    )
    parser.add_argument(
        "suffix",
        type=str,
        help="Branch suffix used in gh_auto_pr.py (must match what was used there)",
    )
    parser.add_argument(
        "--trigger-phrase",
        type=str,
        required=True,
        help="Phrase to comment to trigger",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pr_reviews"),
        help="Directory to save review results (default: results/pr_reviews)",
    )
    parser.add_argument(
        "--max-wait-minutes",
        type=int,
        default=30,
        help="Maximum time to wait for review activity in minutes (default: 30)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Number of parallel workers for processing reviews (default: 1)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete all previous comments on the PR before posting the trigger comment",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(message)s",
    )

    entries = read_csv_entries(args.csv_file)
    if not entries:
        logging.error("No entries to process.")
        sys.exit(1)

    logging.info(f"Found {len(entries)} entries to process")
    logging.info(f"Using trigger phrase: '{args.trigger_phrase}'")
    logging.info(f"Output directory: {args.output_dir}")
    logging.info(f"Using {args.parallel} parallel workers")

    successes = 0

    if args.parallel == 1:
        # Sequential processing
        for i, (url, commit_sha) in enumerate(tqdm(entries, desc="Processing")):
            logging.info(
                f"Processing entry {i+1}/{len(entries)}: {url} with SHA {commit_sha}"
            )

            try:
                ok, msg = process_entry(
                    url,
                    commit_sha,
                    args.suffix,
                    args.trigger_phrase,
                    args.output_dir,
                    args.max_wait_minutes,
                    args.clean,
                )

                if ok:
                    successes += 1
                    print(f"✅ {msg}")
                    logging.info(f"Successfully processed {url} with SHA {commit_sha}")
                else:
                    print(f"❌ {msg}")
                    logging.info(f"Failed to process {url} with SHA {commit_sha}")

            except KeyboardInterrupt:
                print("\nInterrupted by user")
                break
            except Exception as e:
                logging.exception("Error processing %s with SHA %s", url, commit_sha)
                print(f"❌ Error processing {url} with SHA {commit_sha}: {str(e)}")
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=args.parallel) as executor:
            # Submit all tasks
            future_to_entry = {}
            for i, (url, commit_sha) in enumerate(entries):
                future = executor.submit(
                    process_entry,
                    url,
                    commit_sha,
                    args.suffix,
                    args.trigger_phrase,
                    args.output_dir,
                    args.max_wait_minutes,
                    args.clean,
                )
                future_to_entry[future] = (i + 1, url, commit_sha)

            # Process completed tasks with progress bar
            with tqdm(total=len(entries), desc="Processing") as pbar:
                try:
                    for future in as_completed(future_to_entry):
                        entry_num, url, commit_sha = future_to_entry[future]

                        try:
                            ok, msg = future.result()

                            if ok:
                                successes += 1
                                print(f"✅ [{entry_num}/{len(entries)}] {msg}")
                                logging.info(
                                    f"Successfully processed {url} with SHA {commit_sha}"
                                )
                            else:
                                print(f"❌ [{entry_num}/{len(entries)}] {msg}")
                                logging.info(
                                    f"Failed to process {url} with SHA {commit_sha}"
                                )

                        except Exception as e:
                            logging.exception(
                                "Error processing %s with SHA %s", url, commit_sha
                            )
                            print(
                                f"❌ [{entry_num}/{len(entries)}] Error processing {url} with SHA {commit_sha}: {str(e)}"
                            )

                        pbar.update(1)

                except KeyboardInterrupt:
                    print("\nInterrupted by user, cancelling remaining tasks...")
                    # Cancel remaining futures
                    for future in future_to_entry:
                        future.cancel()
                    # Exit the executor context, which will wait for running tasks to complete
                    return

    print("\n==== SUMMARY ====")
    print(f"Processed : {len(entries)}")
    print(f"Successful: {successes}")
    logging.info(
        f"Final results: processed {len(entries)} entries, successfully collected {successes} reviews"
    )


if __name__ == "__main__":
    main()
