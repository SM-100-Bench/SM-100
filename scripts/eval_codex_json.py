#!/usr/bin/env python3
import argparse
import concurrent.futures
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from tqdm import tqdm  # progress bar

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    HEADERS,
    DIFF_CACHE_DIR,
    github_get,
    get_cached_diff,
    save_diff_to_cache,
    get_pr_base_and_line_map,
    get_commit_parent_and_line_map,
    _parse_patch_line_map,
    _merge_line_maps,
)


###############################################################################
# Core processing                                                             #
###############################################################################


###############################################################################
# Core processing                                                             #
###############################################################################


def load_codex_results(codex_results_dir: Path, identifier: str) -> List[Dict]:
    """
    Load Codex results from a .txt file.

    Args:
        codex_results_dir: Directory containing Codex results
        identifier: Identifier for the repository (e.g., owner_repo_pr_number)

    Returns:
        List of issues found by Codex
    """
    # Look for a .txt file with the identifier
    result_file = codex_results_dir / f"{identifier}.txt"
    if not result_file.exists():
        raise FileNotFoundError(f"No Codex results found for {identifier}")

    with open(result_file, "r") as f:
        try:
            # Load the JSON content from the .txt file
            data = json.load(f)

            # Extract the "issues" property from the JSON object
            if "issues" in data and isinstance(data["issues"], list):
                return data["issues"]
            else:
                logging.warning(
                    f"No 'issues' property found in {identifier}.txt or it's not a list"
                )
                return []
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse {identifier}.txt as JSON")
            return []
        except Exception as e:
            logging.warning(f"Error processing {identifier}.txt: {str(e)}")
            return []


def write_issues_to_csv(
    issues: List[Dict],
    repo_url: str,
    repo_id: str,
    csv_file: Path,
    write_header: bool = False,
) -> None:
    """
    Write issues to a CSV file.

    Args:
        issues: List of issues found by Codex
        repo_url: URL of the repository
        repo_id: Identifier for the repository (e.g., owner_repo_pr_number)
        csv_file: Path to the CSV file
        write_header: Whether to write the header row (default: False)

    Returns:
        None
    """
    # Create parent directory if it doesn't exist
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine if we need to write the header (if file doesn't exist or write_header is True)
    file_exists = csv_file.exists()
    write_header = write_header or not file_exists

    # Write issues to CSV
    with open(csv_file, "a" if file_exists else "w", newline="") as f:
        writer = csv.writer(f)
        # Write header if needed
        if write_header:
            writer.writerow(
                [
                    "Repo URL",
                    "Repo ID",
                    "Issue File Name",
                    "Issue File Line",
                    "Issue Description",
                ]
            )

        # Write issues
        for issue in issues:
            file_path = issue.get("file", "")
            line_num = issue.get("line", 0)
            description = issue.get("description", "No description")

            writer.writerow([repo_url, repo_id, file_path, line_num, description])

    logging.info(f"Appended {len(issues)} issues to CSV file: {csv_file}")


def process_url(
    url: str,
    codex_results_dir: Path,
    csv_file: Path = None,
    write_header: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single PR or commit URL.

    Args:
        url: GitHub PR or commit URL
        codex_results_dir: Directory containing Codex results
        csv_file: Path to the CSV file (default: None)
        write_header: Whether to write the header row (default: False)

    Returns:
        Tuple of (overlap_found, message)
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
    logging.info(f"Identified repository: {repo_full}")

    # Determine identifier based on the URL
    if pr_match:
        pr_number = pr_match["number"]
        identifier = f"{owner}_{repo}_{pr_number}"
        logging.info(f"Processing as PR #{pr_number}")
        base_sha, _, gold_map = get_pr_base_and_line_map(owner, repo, int(pr_number))
        # Create URL to the base commit tree
        repo_url = f"https://github.com/{repo_full}/tree/{base_sha}"
        logging.info(f"Base commit tree URL: {repo_url}")
    else:
        commit_sha = commit_match["sha"]
        identifier = f"{owner}_{repo}_{commit_sha}"
        logging.info(f"Processing as commit {commit_sha}")
        parent_sha, gold_map = get_commit_parent_and_line_map(owner, repo, commit_sha)
        # Create URL to the parent commit tree
        repo_url = f"https://github.com/{repo_full}/tree/{parent_sha}"
        logging.info(f"Parent commit tree URL: {repo_url}")

    if not gold_map:
        logging.error(f"No changes found in {url}")
        return False, f"No changes found in {url}"

    for file, lines in gold_map.items():
        for line in lines.copy():
            for i in range(1, 6):
                gold_map[file].add(line + i)
                gold_map[file].add(line - i)

    # Load Codex results
    try:
        issues = load_codex_results(codex_results_dir, identifier)
    except FileNotFoundError:
        logging.error(f"No Codex results found for {identifier}")
        return False, f"No Codex results found for {identifier}"
    except Exception as e:
        logging.error(f"Error loading Codex results: {str(e)}")
        return False, f"Error loading Codex results: {str(e)}"

    # Write issues to CSV if csv_file is provided
    if csv_file and issues:
        write_issues_to_csv(issues, repo_url, identifier, csv_file, write_header)

    # Analyze results for overlap with gold standard
    logging.info(f"Analyzing Codex results for overlap with gold standard")
    found = False
    details: List[str] = []
    issue_count = 0

    for issue in issues:
        issue_count += 1
        file_path = issue.get("file", "")
        line_num = issue.get("line", 0)
        description = issue.get("description", "No description")

        logging.debug(f"Analyzing issue: {description} in {file_path}:{line_num}")

        # Check if the file is in the gold_map
        if file_path in gold_map:
            # Check if the line number is in the set of modified lines
            if line_num in gold_map[file_path]:
                found = True
                overlap_info = (
                    f"  Issue '{description}' overlaps '{file_path}' line {line_num}"
                )
                logging.info(f"Found overlap: {overlap_info}")
                details.append(overlap_info)
            else:
                logging.debug(
                    f"No line overlap found for issue in {file_path}:{line_num}"
                )
        else:
            logging.debug(f"No file overlap found for issue in {file_path}")

    logging.info(f"Analysis complete: processed {issue_count} issues")
    logging.info(f"Overlap detected: {found}")

    summary = "✅ Overlap detected." if found else "❌ No overlap."
    message = f"{url}\n{summary}"
    if details:
        message += "\n" + "\n".join(details)
    return found, message


###############################################################################
# CLI                                                                         #
###############################################################################


def read_urls(path: Path) -> Iterable[str]:
    logging.info(f"Reading URLs from {path}")
    with path.open() as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line.strip()


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(description="Codex bug-fix proximity evaluation.")
    parser.add_argument("url_file", type=Path, help="File with PR/commit URLs.")
    parser.add_argument(
        "--codex-results-dir",
        type=Path,
        default=Path("codex_results"),
        help="Directory containing Codex results (default: codex_results)",
    )
    parser.add_argument(
        "--diff-cache-dir",
        type=Path,
        default=Path("diff_cache"),
        help="Directory to cache diffs (default: diff_cache)",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        default=Path("codex_issues.csv"),
        help="Path to the CSV file for all issues (default: codex_issues.csv)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    # Set the diff cache directory
    global DIFF_CACHE_DIR
    DIFF_CACHE_DIR = args.diff_cache_dir
    DIFF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Get CSV file path
    csv_file = args.csv_file
    # Create parent directory if it doesn't exist
    csv_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(message)s",
    )

    urls = list(read_urls(args.url_file))
    if not urls:
        logging.error("No URLs to process.")
        sys.exit(1)

    # Check if codex results directory exists
    codex_results_dir = args.codex_results_dir
    if not codex_results_dir.exists():
        logging.error(f"Codex results directory not found: {codex_results_dir}")
        sys.exit(1)

    successes = 0
    total_issues_written = 0

    for i, u in enumerate(tqdm(urls, desc="Processing")):
        logging.info(f"Processing URL {i+1}/{len(urls)}: {u}")
        try:
            # Write header only for the first URL
            write_header = i == 0
            ok, msg = process_url(u, codex_results_dir, csv_file, write_header)
            print(msg)
            if ok:
                successes += 1
                logging.info(f"Successfully found overlap for {u}")
            else:
                logging.info(f"No overlap found for {u}")

            # Count issues written to CSV
            if csv_file.exists():
                # Count lines in CSV file (minus header)
                with open(csv_file, "r") as f:
                    total_issues_written = (
                        sum(1 for _ in f) - 1
                    )  # Subtract 1 for header
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception(f"Error processing {u}: {str(e)}")

    print("\n==== SUMMARY ====")
    print(f"Processed : {len(urls)}")
    print(f"Overlap   : {successes}")
    print(f"Issues    : {total_issues_written}")
    print(f"CSV File  : {csv_file}")
    logging.info(
        f"Final results: processed {len(urls)} URLs, found overlap in {successes}, wrote {total_issues_written} issues to {csv_file}"
    )


if __name__ == "__main__":
    main()
