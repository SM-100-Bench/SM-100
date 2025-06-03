import argparse
import concurrent.futures
import logging
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

from git import Repo  # type: ignore
import requests
from tqdm import tqdm  # progress bar

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    HEADERS,
    DIFF_CACHE_DIR,
    get_result_filename,
    github_get,
    get_pr_base_and_line_map,
    get_commit_parent_and_line_map,
    _parse_patch_line_map,
    _merge_line_maps,
    save_scan_results,
    result_exists,
    load_scan_results,
)


DEVIN_API_KEY = os.getenv("DEVIN_API_KEY")
if not DEVIN_API_KEY:
    raise RuntimeError("DEVIN_API_KEY environment variable must be set.")


###############################################################################
# Devin API integration                                                       #
###############################################################################


###############################################################################
# Core processing                                                             #
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


def run_devin(repo_id, subsystems):
    """
    Run Devin on a repository with specified subsystems.

    Args:
        repo_id (str): Repository ID in the format {repo_owner}_{repo_name}_{pr_number_or_commit}
        subsystems (list): List of subsystems, each a dict with "name" and "files" keys

    Returns:
        object: The scan results from Devin
    """
    logging.info(f"Running Devin for repo ID: {repo_id}")

    # Function to wait for session to be blocked (ready for next input)
    def wait_for_blocked_status(session_id, max_retries=20, retry_interval=60):
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
            import time

            time.sleep(retry_interval)

        return None

    # Check if a session with this title already exists
    sessions = list_devin_sessions()
    existing_session = next((s for s in sessions if s.get("title") == repo_id), None)

    if existing_session:
        logging.info(f"Found existing Devin session with title '{repo_id}'")
        session_id = existing_session["session_id"]

        # Fetch the session details to analyze messages and determine which subsystems have been processed
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

        # Check if the session is in a state where we can resume
        status = session_data.get("status_enum")
        if status != "blocked":
            logging.warning(
                f"Session {session_id} is not in a blocked state (status: {status}), cannot resume"
            )

            # Just return the current structured output
            structured_output = session_data.get("structured_output", {})
            issues = structured_output.get("issues", [])
            logging.info(f"Found {len(issues)} issues in existing session")

            return {
                "session_id": session_id,
                "issues": issues,
            }

        # Analyze messages to determine which subsystems have been processed
        messages = session_data.get("messages", [])
        processed_subsystems = set()

        # Extract the first subsystem from the initial prompt
        initial_prompt = next(
            (m["message"] for m in messages if m.get("type") == "user_message"), ""
        )
        if "Start by reviewing the" in initial_prompt:
            # Extract the subsystem name from the prompt
            # This is a simple heuristic and might need to be adjusted based on the actual prompt format
            start_idx = initial_prompt.find("Start by reviewing the") + len(
                "Start by reviewing the"
            )
            end_idx = initial_prompt.find("subsystem", start_idx)
            if start_idx > 0 and end_idx > start_idx:
                first_subsystem_name = initial_prompt[start_idx:end_idx].strip()
                processed_subsystems.add(first_subsystem_name)
                logging.info(
                    f"Detected first subsystem '{first_subsystem_name}' from initial prompt"
                )

        # Extract subsystems from subsequent messages
        for msg in messages:
            if msg.get(
                "type"
            ) == "user_message" and "Please review the code in the" in msg.get(
                "message", ""
            ):
                message = msg.get("message", "")
                start_idx = message.find("Please review the code in the") + len(
                    "Please review the code in the"
                )
                end_idx = message.find("subsystem", start_idx)
                if start_idx > 0 and end_idx > start_idx:
                    subsystem_name = message[start_idx:end_idx].strip()
                    processed_subsystems.add(subsystem_name)
                    logging.info(
                        f"Detected processed subsystem '{subsystem_name}' from messages"
                    )

        logging.info(
            f"Found {len(processed_subsystems)} already processed subsystems: {processed_subsystems}"
        )

        # Get the current structured output
        structured_output = session_data.get("structured_output", {})
        all_issues = structured_output.get("issues", [])
        logging.info(f"Found {len(all_issues)} issues in existing session")

        # Determine which subsystems still need to be processed
        remaining_subsystems = []
        for subsystem in subsystems:
            subsystem_name = subsystem["name"]
            if subsystem_name not in processed_subsystems:
                remaining_subsystems.append(subsystem)

        if not remaining_subsystems:
            logging.info("All subsystems have already been processed")
            return {
                "session_id": session_id,
                "issues": all_issues,
            }

        logging.info(f"Resuming with {len(remaining_subsystems)} remaining subsystems")

        # Process the remaining subsystems
        for i, subsystem in enumerate(remaining_subsystems):
            subsystem_name = subsystem["name"]
            files = ", ".join([f"'{file}'" for file in subsystem["files"]])

            logging.info(
                f"Processing subsystem {i+1}/{len(remaining_subsystems)}: {subsystem_name}"
            )

            # Send a message about this subsystem
            message = f"""Please review the code in the {subsystem_name} subsystem (consisting of {files}) within the {repo_id} repository.
Remember to update the structured output with any issues you find."""

            message_url = f"https://api.devin.ai/v1/session/{session_id}/message"
            payload = {"message": message}
            headers = {
                "Authorization": f"Bearer {DEVIN_API_KEY}",
                "Content-Type": "application/json",
            }

            logging.info(f"Sending message about subsystem {subsystem_name}")
            response = requests.request(
                "POST", message_url, json=payload, headers=headers
            )
            if response.status_code != 200:
                logging.error(
                    f"Failed to send message: {response.status_code} {response.text}"
                )
                raise Exception("Failed to send message")

            # Wait for the session to be blocked again (ready for next input)
            session_data = wait_for_blocked_status(session_id)

            # Extract any issues found for this subsystem
            structured_output = session_data.get("structured_output", {})
            issues = structured_output.get("issues", [])
            logging.info(f"Found {len(issues)} issues in subsystem {subsystem_name}")

            # Update our running list (we don't need to extend since we're getting the full list each time)
            all_issues = issues

        # Return the final scan results
        scan_results = {
            "session_id": session_id,
            "issues": all_issues,
        }

        logging.info(f"Total issues found across all subsystems: {len(all_issues)}")
        return scan_results
    else:
        logging.info(f"Creating new Devin session with title '{repo_id}'")

        # Create a new session with the first subsystem included in the initial prompt
        url = "https://api.devin.ai/v1/sessions"

        # Get the first subsystem details
        first_subsystem = subsystems[0]
        first_subsystem_name = first_subsystem["name"]
        first_subsystem_files = ", ".join(
            [f"'{file}'" for file in first_subsystem["files"]]
        )

        # Initial prompt that includes the first subsystem
        prompt = f"""Please review the code in the {repo_id} repository for potential bugs.
Focus on identifying issues that represent objectively incorrect behavior, could lead to exceptions or program crashes, or constitute security vulnerabilities.

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

Start by reviewing the {first_subsystem_name} subsystem (consisting of {first_subsystem_files}).
After you've completed this subsystem, I'll provide you with additional subsystems to analyze.
"""

        payload = {"prompt": prompt, "title": repo_id, "idempotent": True}

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

        # Process each subsystem sequentially
        all_issues = []

        # Wait for the first subsystem to be processed (it was included in the initial prompt)
        session_data = wait_for_blocked_status(session_id)

        # Extract any issues found for the first subsystem
        structured_output = session_data.get("structured_output", {})
        issues = structured_output.get("issues", [])
        logging.info(f"Found {len(issues)} issues in subsystem {subsystems[0]['name']}")
        all_issues.extend(issues)

        # Process the remaining subsystems
        for i, subsystem in enumerate(subsystems[1:], 1):
            subsystem_name = subsystem["name"]
            files = ", ".join([f"'{file}'" for file in subsystem["files"]])

            logging.info(
                f"Processing subsystem {i+1}/{len(subsystems)}: {subsystem_name}"
            )

            # Send a message about this subsystem
            message = f"""Please review the code in the {subsystem_name} subsystem (consisting of {files}) within the {repo_id} repository.
Remember to update the structured output with any issues you find."""

            message_url = f"https://api.devin.ai/v1/session/{session_id}/message"
            payload = {"message": message}
            headers = {
                "Authorization": f"Bearer {DEVIN_API_KEY}",
                "Content-Type": "application/json",
            }

            logging.info(f"Sending message about subsystem {subsystem_name}")
            response = requests.request(
                "POST", message_url, json=payload, headers=headers
            )
            if response.status_code != 200:
                logging.error(
                    f"Failed to send message: {response.status_code} {response.text}"
                )
                raise Exception("Failed to send message")

            # Wait for the session to be blocked again (ready for next input)
            session_data = wait_for_blocked_status(session_id)

            # Extract any issues found for this subsystem
            structured_output = session_data.get("structured_output", {})
            issues = structured_output.get("issues", [])
            logging.info(f"Found {len(issues)} issues in subsystem {subsystem_name}")

            # Add these issues to our running list
            all_issues.extend(issues)

        # Create the final scan results
        scan_results = {
            "session_id": session_id,
            "issues": all_issues,
        }

        logging.info(f"Total issues found across all subsystems: {len(all_issues)}")
        return scan_results


def process_url(
    url: str,
    results_dir: Path,
    resume: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single PR or commit URL.

    If resume=True and results already exist for this URL, they will be loaded
    from disk instead of reprocessing the URL.
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

    if pr_match:
        logging.info(f"Processing as PR #{pr_match['number']}")
        _, _, gold_map = get_pr_base_and_line_map(owner, repo, int(pr_match["number"]))
    else:
        sha = commit_match["sha"]
        logging.info(f"Processing as commit {sha}")
        _, gold_map = get_commit_parent_and_line_map(owner, repo, sha)

    if not gold_map:
        logging.error(f"No changes found in {url}")
        return False, f"No changes found in {url}"

    scan_res = None
    if resume and result_exists(url, results_dir, prefix="fix_scan"):
        logging.info(f"Found existing results for {url}, attempting to load")
        try:
            with open(
                get_result_filename(url, results_dir, prefix="fix_scan"), "r"
            ) as f:
                scan_res = json.load(f)
        except:
            logging.warning(
                f"Failed to load existing results for {url}, re-scanning", exc_info=True
            )

    if scan_res is None:
        logging.info(f"Running Devin scan on repository")

        # Determine repo_id based on the URL
        if pr_match:
            repo_id = f"{owner}_{repo}_{pr_match['number']}"
        else:
            repo_id = f"{owner}_{repo}_{commit_match['sha']}"

        logging.info(f"Using repo ID: {repo_id}")

        # Load subsystems from JSON file
        subsystems_file = Path(f"subsystems/{repo_id}.json")

        logging.info(f"Loading subsystems from {subsystems_file}")
        import json

        with open(subsystems_file, "r") as f:
            subsystems = json.load(f)

        # Filter subsystems to only include those with files that overlap with gold_map
        filtered_subsystems = [
            s for s in subsystems if set(s["files"]) & set(gold_map.keys())
        ]

        if filtered_subsystems:
            logging.info(
                f"Filtered from {len(subsystems)} to {len(filtered_subsystems)} subsystems with overlapping files"
            )
            subsystems = filtered_subsystems
        else:
            logging.warning(
                "No subsystems with overlapping files found, using all subsystems"
            )

        # Run Devin with the repo_id and filtered subsystems
        scan_res = run_devin(repo_id, subsystems)

    # Save both the scan results and our analysis
    logging.info(f"Saving scan results to {results_dir}")
    with open(get_result_filename(url, results_dir, prefix="fix_scan"), "w") as f:
        json.dump(scan_res, f)

    logging.info(f"Analyzing scan results for overlap with gold standard")
    found = False
    details: List[str] = []
    issue_count = 0

    # Get the issues from the scan results
    issues = scan_res.get("issues", [])

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
    parser = argparse.ArgumentParser(description="Bug-fix proximity evaluation.")
    parser.add_argument("url_file", type=Path, help="File with PR/commit URLs.")
    parser.add_argument("--tmpdir", type=Path, help="Directory to store clones.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("devin_results"),
        help="Directory to store scan results (default: devin_results)",
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
        help="Resume from previous run, skipping already processed URLs",
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
        help="Randomly shuffle the URLs before processing",
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

    urls = list(read_urls(args.url_file))
    if not urls:
        logging.error("No URLs to process.")
        sys.exit(1)

    # Shuffle URLs if requested
    if args.shuffle:
        logging.info(f"Shuffling {len(urls)} URLs")
        random.shuffle(urls)

    # Create results directory
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving scan results to {results_dir}")

    # Function to process a single URL for parallel execution
    def process_url_wrapper(url_index_tuple):
        index, url = url_index_tuple
        logging.info(f"Processing URL {index+1}/{len(urls)}: {url}")
        try:
            ok, msg = process_url(url, results_dir, args.resume)
            return ok, msg, url, None  # None means no exception
        except KeyboardInterrupt:
            return False, "Interrupted", url, "KeyboardInterrupt"
        except Exception as e:
            logging.exception(f"Error processing {url}")
            return False, f"Error: {str(e)}", url, str(e)

    successes = 0

    # Use parallel processing if requested
    if args.parallel > 1:
        logging.info(f"Using parallel processing with {args.parallel} workers")

        # Create a progress bar that will be updated as tasks complete
        with tqdm(total=len(urls), desc="Processing") as progress_bar:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.parallel
            ) as executor:
                # Submit all tasks
                future_to_url = {
                    executor.submit(process_url_wrapper, (i, u)): (i, u)
                    for i, u in enumerate(urls)
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_url):
                    i, u = future_to_url[future]
                    try:
                        ok, msg, url, exception = future.result()
                        print(msg)
                        if ok:
                            successes += 1
                            logging.info(f"Successfully found overlap for {url}")
                        else:
                            if exception:
                                logging.error(f"Error processing {url}: {exception}")
                            else:
                                logging.info(f"No overlap found for {url}")
                    except Exception as e:
                        logging.exception(f"Error getting result for {u}")

                    # Update progress bar
                    progress_bar.update(1)
    else:
        # Sequential processing (original behavior)
        for i, u in enumerate(tqdm(urls, desc="Processing")):
            logging.info(f"Processing URL {i+1}/{len(urls)}: {u}")
            try:
                ok, msg = process_url(u, results_dir, args.resume)
                print(msg)
                if ok:
                    successes += 1
                    logging.info(f"Successfully found overlap for {u}")
                else:
                    logging.info(f"No overlap found for {u}")
            except KeyboardInterrupt:
                break
            except:
                logging.exception("Error processing %s", u)

    print("\n==== SUMMARY ====")
    print(f"Processed : {len(urls)}")
    print(f"Overlap   : {successes}")
    logging.info(
        f"Final results: processed {len(urls)} URLs, found overlap in {successes}"
    )


if __name__ == "__main__":
    main()
