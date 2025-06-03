#!/usr/bin/env python3
"""
Script to construct PR review prompts and save them to files.
Usage:
  python construct_pr_prompts.py prs.csv output_directory
"""

import argparse
import csv
import json
import logging
import os
import re
import requests
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    get_pr_base_and_line_map,
    get_commit_parent_and_line_map,
    get_cached_diff,
    save_diff_to_cache,
    github_get,
    DIFF_CACHE_DIR,
    HEADERS,
)


def read_csv_entries(csv_path: Path) -> List[Tuple[str, str]]:
    """
    Read URL and commit entries from CSV file.

    Args:
        csv_path (Path): Path to CSV file with format: url,commit_sha

    Returns:
        List[Tuple[str, str]]: List of (url, commit_sha) tuples
    """
    logging.info(f"Reading CSV entries from {csv_path}")
    entries = []

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)

        # Skip header if it exists
        first_row = next(reader, None)
        if first_row and first_row[0].lower() == "url":
            logging.info("Skipping header row")
        else:
            # First row is data, add it back
            if first_row and len(first_row) >= 2:
                entries.append((first_row[0].strip(), first_row[1].strip()))

        # Read the rest of the rows
        for row in reader:
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                url = row[0].strip()
                commit_sha = row[1].strip()
                entries.append((url, commit_sha))
            else:
                logging.warning(f"Skipping invalid row: {row}")

    logging.info(f"Found {len(entries)} valid entries in CSV")
    return entries


def fetch_commit_diff(owner: str, repo: str, commit_sha: str) -> str:
    """
    Fetch the diff for a specific commit, using cache if available.

    Args:
        owner (str): Repository owner
        repo (str): Repository name
        commit_sha (str): Commit SHA

    Returns:
        str: The diff text
    """
    # Create a cache key for this commit
    cache_key = f"intro_commit_{owner}_{repo}_{commit_sha}"

    # Check if we have a cached diff
    diff_text = get_cached_diff(cache_key)

    if diff_text is None:
        logging.info(f"Fetching commit diff for {owner}/{repo} commit {commit_sha}")
        api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_sha}"

        try:
            response = requests.get(
                api_url,
                headers=HEADERS | {"Accept": "application/vnd.github.diff"},
            )
            if response.status_code != 200:
                logging.error(
                    f"Failed to fetch commit diff: {response.status_code} {response.text}"
                )
                raise Exception("Failed to fetch commit diff")

            diff_text = response.text
            # Save the diff to cache
            save_diff_to_cache(cache_key, diff_text)
        except Exception as e:
            logging.error(f"Error fetching diff: {str(e)}")
            raise

    return diff_text


def get_safe_filename(identifier: str, prefix: str = "prompt") -> str:
    """
    Generate a consistent filename for prompts based on identifier.

    Args:
        identifier: A unique identifier (e.g., URL or commit SHA)
        prefix: Prefix for the filename

    Returns:
        A safe filename
    """
    # Create a safe filename from the identifier
    safe_id = re.sub(r"[^\w\-]", "_", identifier)
    if len(safe_id) > 100:  # Avoid excessively long filenames
        safe_id = safe_id[:100]

    return f"{prefix}_{safe_id}.txt"


def parse_repo_id(repo_id: str) -> Tuple[str, str, str, bool]:
    """
    Parse a repository ID in the format {repo_owner}_{repo_name}_{pr_number_or_commit}

    Returns:
        Tuple of (owner, repo, pr_number_or_commit, is_pr)
    """
    parts = repo_id.split("_", 2)
    if len(parts) < 3:
        raise ValueError(f"Invalid repo_id format: {repo_id}")

    owner, repo, identifier = parts

    # Check if identifier is a PR number (numeric) or commit SHA (alphanumeric)
    is_pr = identifier.isdigit()

    return owner, repo, identifier, is_pr


def construct_prompt(repo_id: str, diff_text: str) -> str:
    """
    Construct a prompt for code review based on the repository ID and diff text.

    Args:
        repo_id: Repository ID in the format {repo_owner}_{repo_name}_{pr_number_or_commit}
        diff_text: The diff text for the commit

    Returns:
        The constructed prompt
    """
    return f"""Please review the following commit for potential bugs:

```
{diff_text}
```

Focus on identifying issues that represent objectively incorrect behavior, could lead to exceptions or program crashes, or constitute security vulnerabilities.

Report all of your findings in a single JSON object with the following format:

{{
  "issues": [
    {{
      "file": "src/App.tsx",
      "line": 42,
      "description": "Memory leak in useEffect cleanup"
    }}
  ]
}}"""


def main():
    parser = argparse.ArgumentParser(
        description="Construct prompts as done in eval_codex.py and save them to files."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="CSV file with url,commit_sha format for PR/commit analysis",
    )
    parser.add_argument("output_dir", type=Path, help="Directory to save prompt files")
    parser.add_argument(
        "--diff-cache-dir",
        type=Path,
        default=Path("diff_cache"),
        help="Directory to cache diffs (default: diff_cache)",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0, help="Increase verbosity"
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG

    logging.basicConfig(
        level=log_level,
        format="%(levelname)s: %(message)s",
    )

    # Set the diff cache directory
    global DIFF_CACHE_DIR
    DIFF_CACHE_DIR = args.diff_cache_dir
    DIFF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Create output directory if it doesn't exist
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process entries from CSV
    try:
        entries = read_csv_entries(args.csv_file)
        logging.info(f"Processing {len(entries)} entries from CSV file")

        for url, commit_sha in entries:
            try:
                # Parse the URL to extract repo info
                pr_match = PR_URL_RE.fullmatch(url.strip())
                commit_match = COMMIT_URL_RE.fullmatch(url.strip())
                if not (pr_match or commit_match):
                    logging.error(f"URL format not recognized: {url}")
                    continue

                owner = pr_match["owner"] if pr_match else commit_match["owner"]
                repo = pr_match["repo"] if pr_match else commit_match["repo"]
                repo_id = f"{owner}_{repo}_{commit_sha[:8]}"

                logging.info(f"Processing {url} with commit {commit_sha}")

                # Fetch the diff for the commit
                try:
                    diff_text = fetch_commit_diff(owner, repo, commit_sha)
                    logging.info(f"Successfully fetched diff for {commit_sha}")
                except Exception as e:
                    logging.error(f"Failed to fetch diff: {str(e)}")
                    continue

                # Construct the prompt with the diff
                prompt = construct_prompt(repo_id, diff_text)

                # Generate output filename
                output_filename = get_safe_filename(f"{url}_{commit_sha[:8]}")
                output_path = args.output_dir / output_filename

                # Save prompt to file
                with open(output_path, "w") as f:
                    f.write(prompt)

                logging.info(f"Created prompt for {url} -> {output_path}")

            except Exception as e:
                logging.error(f"Error processing {url}: {str(e)}", exc_info=True)
    except Exception as e:
        logging.error(f"Error reading CSV file: {str(e)}", exc_info=True)

    logging.info(f"All prompts have been saved to {args.output_dir}")


if __name__ == "__main__":
    main()
