import argparse
import concurrent.futures
import csv
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import requests
from tqdm import tqdm

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    HEADERS,
    DIFF_CACHE_DIR,
    get_result_filename,
    github_get,
    get_pr_base_and_line_map,
    get_commit_parent_and_line_map,
    result_exists,
)


DEVIN_API_KEY = os.getenv("DEVIN_API_KEY")
if not DEVIN_API_KEY:
    raise RuntimeError("DEVIN_API_KEY environment variable must be set.")


###############################################################################
# Devin API integration                                                       #
###############################################################################

# Variable to store Devin sessions to avoid listing them multiple times
_devin_sessions = None


def list_devin_sessions():
    """
    List all existing Devin sessions.

    Returns:
        list: A list of session objects
    """
    global _devin_sessions

    if _devin_sessions is None:
        url = "https://api.devin.ai/v1/sessions"
        headers = {"Authorization": f"Bearer {DEVIN_API_KEY}"}

        response = requests.request("GET", url, headers=headers)
        if response.status_code != 200:
            logging.error(
                f"Failed to list Devin sessions: {response.status_code} {response.text}"
            )
            raise Exception("Failed to list Devin sessions")

        _devin_sessions = response.json()
        if "sessions" in _devin_sessions:
            _devin_sessions = _devin_sessions["sessions"]
        logging.info(f"Found {len(_devin_sessions)} existing Devin sessions")

    return _devin_sessions


def run_devin_pr_analysis(repo_url: str, commit_sha: str):
    """
    Run Devin to analyze a specific PR/commit for bugs.

    Args:
        repo_url (str): Repository URL (e.g., "owner/repo")
        commit_sha (str): Commit SHA to analyze

    Returns:
        object: The scan results from Devin
    """
    logging.info(
        f"Running Devin PR analysis for repo: {repo_url}, commit: {commit_sha}"
    )

    # Create a unique session title
    session_title = f"pr_analysis_{repo_url.replace('/', '_')}_{commit_sha[:8]}"

    # Function to wait for session to be blocked (ready for next input)
    def wait_for_blocked_status(session_id, max_retries=30, retry_interval=60):
        session_url = f"https://api.devin.ai/v1/session/{session_id}"
        headers = {"Authorization": f"Bearer {DEVIN_API_KEY}"}

        for attempt in range(max_retries):
            response = requests.request("GET", session_url, headers=headers)
            if response.status_code != 200:
                logging.error(
                    f"Failed to fetch session details: {response.status_code} {response.text}"
                )
                raise Exception("Failed to fetch session details")

            session_data = response.json()
            status = session_data.get("status_enum")

            if status == "blocked":
                logging.info(
                    f"Session {session_id} is blocked and ready for next input"
                )
                return session_data

            # If we've reached the maximum number of retries, break out of the loop
            if attempt == max_retries - 1:
                logging.warning(
                    f"Timeout waiting for session {session_id} to be blocked"
                )
                return session_data

            # Wait before trying again
            logging.info(
                f"Session {session_id} is still {status}, waiting {retry_interval} seconds..."
            )
            time.sleep(retry_interval)

        return None

    # Check if a session with this title already exists
    sessions = list_devin_sessions()
    existing_session = next(
        (s for s in sessions if s.get("title") == session_title), None
    )

    if existing_session:
        logging.info(f"Found existing Devin session with title '{session_title}'")
        session_id = existing_session["session_id"]

        # Fetch the session details to get current results
        logging.info(f"Fetching details for existing session {session_id}")
        url = f"https://api.devin.ai/v1/session/{session_id}"
        headers = {"Authorization": f"Bearer {DEVIN_API_KEY}"}

        response = requests.request("GET", url, headers=headers)
        if response.status_code != 200:
            logging.error(
                f"Failed to fetch session details: {response.status_code} {response.text}"
            )
            raise Exception("Failed to fetch session details")

        session_data = response.json()

        # Get the current structured output
        structured_output = session_data.get("structured_output", {})
        issues = structured_output.get("issues", [])
        logging.info(f"Found {len(issues)} issues in existing session")

        return {
            "session_id": session_id,
            "issues": issues,
        }
    else:
        logging.info(f"Creating new Devin session with title '{session_title}'")

        # Create a new session for PR/commit analysis
        url = "https://api.devin.ai/v1/sessions"

        # Create the prompt for analyzing a specific commit
        prompt = f"""Please analyze the repository {repo_url} for potential bugs introduced in commit {commit_sha}.

Your task is to:
1. Clone the repository {repo_url}
2. Examine the specific changes made in commit {commit_sha}
3. Identify potential bugs, issues, or problems in the changed code. Focus on issues that represent objectively incorrect behavior, could lead to exceptions or program crashes, or constitute security vulnerabilities

Update the structured output immediately upon discovering new issues. The output should conform to the following JSON format:

{{
  "issues": [
    {{
      "file": "src/App.tsx",
      "line": 42,
      "description": "Memory leak in useEffect cleanup"
    }}
  ]
}}

Please be thorough in your analysis and examine:
- Logic errors in the changed code
- Potential null pointer dereferences
- Resource leaks
- Security vulnerabilities
- Race conditions
- Incorrect error handling
- Type mismatches or casting issues
- Boundary condition errors

Focus specifically on the changes introduced in commit {commit_sha}, but also consider how these changes interact with the existing codebase.
"""

        payload = {"prompt": prompt, "title": session_title, "idempotent": True}

        headers = {
            "Authorization": f"Bearer {DEVIN_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.request("POST", url, json=payload, headers=headers)
        if response.status_code != 200:
            logging.error(
                f"Failed to create Devin session: {response.status_code} {response.text}"
            )
            raise Exception("Failed to create Devin session")

        session_data = response.json()
        session_id = session_data["session_id"]
        logging.info(f"Created new Devin session with ID: {session_id}")

        # Wait for the analysis to complete
        session_data = wait_for_blocked_status(session_id)

        # Extract the issues found
        structured_output = session_data.get("structured_output", {})
        issues = structured_output.get("issues", [])
        logging.info(f"Found {len(issues)} issues in commit analysis")

        # Create the final scan results
        scan_results = {
            "session_id": session_id,
            "issues": issues,
        }

        logging.info(f"Total issues found in commit {commit_sha}: {len(issues)}")
        return scan_results


def process_commit_entry(
    url: str,
    commit_sha: str,
    results_dir: Path,
    resume: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single repository/commit entry.

    Args:
        url (str): GitHub PR or commit URL
        commit_sha (str): Commit SHA to analyze
        results_dir (Path): Directory to store results
        resume (bool): Whether to resume from existing results

    Returns:
        Tuple[bool, str]: (success, message)
    """
    logging.info(f"Processing URL: {url}")

    # Parse the URL to extract repo info
    pr_match = PR_URL_RE.fullmatch(url.strip())
    commit_match = COMMIT_URL_RE.fullmatch(url.strip())
    if not (pr_match or commit_match):
        logging.error(f"URL format not recognized: {url}")
        raise ValueError(f"Unsupported URL format: {url}")

    owner = pr_match["owner"] if pr_match else commit_match["owner"]
    repo = pr_match["repo"] if pr_match else commit_match["repo"]
    repo_full = f"https://github.com/{owner}/{repo}"
    logging.info(f"Identified repository: {repo_full}")

    # Check if we should resume from existing results
    scan_res = None
    if resume and result_exists(url, results_dir, prefix="pr_scan"):
        logging.info(f"Found existing results for {url}, attempting to load")
        try:
            with open(
                get_result_filename(url, results_dir, prefix="pr_scan"), "r"
            ) as f:
                scan_res = json.load(f)
        except Exception:
            logging.warning(
                f"Failed to load existing results for {url}, re-scanning",
                exc_info=True,
            )

    if scan_res is None:
        logging.info(f"Running Devin analysis on {repo_full} commit {commit_sha}")

        # Run Devin analysis
        scan_res = run_devin_pr_analysis(repo_full, commit_sha)

    # Save the scan results
    logging.info(f"Saving scan results to {results_dir}")
    with open(get_result_filename(url, results_dir, prefix="pr_scan"), "w") as f:
        json.dump(scan_res, f, indent=2)

    # Analyze the results
    issues = scan_res.get("issues", [])
    issue_count = len(issues)

    logging.info(f"Analysis complete: found {issue_count} issues")

    # Create summary message
    if issue_count > 0:
        summary = f"✅ Found {issue_count} potential issues."
        details = []
        for issue in issues[:5]:  # Show first 5 issues
            file_path = issue.get("file", "unknown")
            line_num = issue.get("line", "unknown")
            description = issue.get("description", "No description")
            details.append(f"  - {file_path}:{line_num} - {description}")

        if issue_count > 5:
            details.append(f"  ... and {issue_count - 5} more issues")

        message = f"{url}\n{summary}\n" + "\n".join(details)
    else:
        summary = "❌ No issues found."
        message = f"{url}\n{summary}"

    return issue_count > 0, message


###############################################################################
# CSV processing                                                              #
###############################################################################


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


###############################################################################
# CLI                                                                         #
###############################################################################


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(description="PR/Commit bug analysis using Devin.")
    parser.add_argument(
        "csv_file", type=Path, help="CSV file with url,commit_sha format."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("devin_pr_results"),
        help="Directory to store scan results (default: devin_pr_results)",
    )
    parser.add_argument(
        "--diff-cache-dir",
        type=Path,
        default=Path("diff_cache"),
        help="Directory to cache diffs (default: diff_cache)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skipping already processed entries",
    )
    parser.add_argument(
        "--parallel",
        "-p",
        type=int,
        default=1,
        help="Number of parallel processes to use (default: 1, sequential processing)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle the entries before processing",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    # Set the diff cache directory
    global DIFF_CACHE_DIR
    DIFF_CACHE_DIR = args.diff_cache_dir
    DIFF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.WARNING - 10 * min(args.verbose, 2),
        format="%(levelname)s %(message)s",
    )

    # Read CSV entries
    try:
        entries = read_csv_entries(args.csv_file)
    except Exception as e:
        logging.error(f"Failed to read CSV file: {e}")
        sys.exit(1)

    if not entries:
        logging.error("No valid entries found in CSV file.")
        sys.exit(1)

    # Shuffle entries if requested
    if args.shuffle:
        logging.info(f"Shuffling {len(entries)} entries")
        random.shuffle(entries)

    # Create results directory
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving scan results to {results_dir}")

    # Function to process a single entry for parallel execution
    def process_entry_wrapper(entry_index_tuple):
        index, (url, commit_sha) = entry_index_tuple
        logging.info(f"Processing entry {index+1}/{len(entries)}: {url}")
        try:
            found_issues, msg = process_commit_entry(
                url, commit_sha, results_dir, args.resume
            )
            return found_issues, msg, url, None  # None means no exception
        except KeyboardInterrupt:
            return False, "Interrupted", url, "KeyboardInterrupt"
        except Exception as e:
            logging.exception(f"Error processing {url}")
            return False, f"Error: {str(e)}", url, str(e)

    successes = 0
    total_issues = 0

    # Use parallel processing if requested
    if args.parallel > 1:
        logging.info(f"Using parallel processing with {args.parallel} workers")

        # Create a progress bar that will be updated as tasks complete
        with tqdm(total=len(entries), desc="Processing") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel
            ) as executor:
                # Submit all tasks
                future_to_entry = {
                    executor.submit(process_entry_wrapper, (i, entry)): (i, entry)
                    for i, entry in enumerate(entries)
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_entry):
                    i, entry = future_to_entry[future]
                    try:
                        found_issues, msg, url, exception = future.result()
                        print(msg)
                        if found_issues:
                            successes += 1
                            # Count total issues from the message (rough estimate)
                            if "Found" in msg and "issues" in msg:
                                try:
                                    issue_count = int(
                                        msg.split("Found ")[1].split(" ")[0]
                                    )
                                    total_issues += issue_count
                                except (IndexError, ValueError):
                                    pass
                            logging.info(f"Successfully found issues for {url}")
                        else:
                            if exception:
                                logging.error(f"Error processing {url}: {exception}")
                            else:
                                logging.info(f"No issues found for {url}")
                    except Exception as e:
                        logging.exception(f"Error getting result for {entry}")

                    # Update progress bar
                    progress_bar.update(1)
    else:
        # Sequential processing (original behavior)
        for i, (url, commit_sha) in enumerate(tqdm(entries, desc="Processing")):
            logging.info(f"Processing entry {i+1}/{len(entries)}: {url}")
            try:
                found_issues, msg = process_commit_entry(
                    url, commit_sha, results_dir, args.resume
                )
                print(msg)
                if found_issues:
                    successes += 1
                    # Count total issues from the message (rough estimate)
                    if "Found" in msg and "issues" in msg:
                        try:
                            issue_count = int(msg.split("Found ")[1].split(" ")[0])
                            total_issues += issue_count
                        except (IndexError, ValueError):
                            pass
                    logging.info(f"Successfully found issues for {url}")
                else:
                    logging.info(f"No issues found for {url}")
            except KeyboardInterrupt:
                break
            except Exception:
                logging.exception(f"Error processing {url}")

    print("\n==== SUMMARY ====")
    print(f"Processed     : {len(entries)} entries")
    print(f"Found Issues  : {successes} entries")
    print(f"Total Issues  : {total_issues} issues")
    logging.info(
        f"Final results: processed {len(entries)} entries, found issues in {successes} entries, total {total_issues} issues"
    )


if __name__ == "__main__":
    main()
