import argparse
import concurrent.futures
import csv
import json
import logging
import os
import random
import shutil
import subprocess
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
    github_get,
)


###############################################################################
# Claude CLI integration                                                      #
###############################################################################


def run_claude_subprocess(repo_path: Path, prompt: str) -> Dict:
    """
    Run the claude CLI program as a subprocess with the given prompt.

    Args:
        prompt (str): The prompt to send to Claude

    Returns:
        Dict: The parsed JSON response from Claude
    """
    try:
        # Run the claude command with the prompt and JSON output format
        cmd = ["claude", "-p", prompt, "--output-format", "json"]

        logging.info(
            f"Running claude command: {' '.join(cmd[:2])} '<prompt>' {' '.join(cmd[3:])}"
        )

        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(repo_path), timeout=900  # 15 minute timeout
        )

        if result.returncode != 0:
            logging.error(f"Claude command failed with return code {result.returncode}")
            logging.error(f"stderr: {result.stderr}")
            raise Exception(f"Claude command failed: {result.stderr}")

        # Parse the JSON output
        try:
            response_data = json.loads(result.stdout)
            return response_data
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Claude JSON output: {e}")
            logging.error(f"Raw output: {result.stdout}")
            raise Exception(f"Failed to parse Claude JSON output: {e}")

    except subprocess.TimeoutExpired:
        logging.error("Claude command timed out")
        raise Exception("Claude command timed out")
    except FileNotFoundError:
        logging.error(
            "Claude command not found. Make sure 'claude' is installed and in PATH"
        )
        raise Exception("Claude command not found")


###############################################################################
# GitHub diff acquisition                                                     #
###############################################################################


def get_commit_parent_and_line_map(
    owner: str, repo: str, sha: str
) -> Tuple[str, Dict[str, Set[int]]]:
    """
    Return (parent_sha, gold_line_map) for an individual commit.
    """
    logging.info(f"Fetching commit information for {owner}/{repo} commit {sha}")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{sha}"
    info = github_get(api_url)

    parents = info.get("parents", [])
    if not parents:
        logging.error(f"Commit {sha} has no parents")
        raise ValueError("Root commit provided; cannot determine parent.")

    parent_sha = parents[0]["sha"]
    logging.debug(f"Parent commit SHA: {parent_sha}")

    return parent_sha, {}


###############################################################################
# Core processing                                                             #
###############################################################################


def run_claude_on_staged_changes(repo_path: Path, identifier: str) -> Dict:
    """
    Run Claude on staged changes in a repository.

    Args:
        repo_path (Path): Path to the repository with staged changes
        identifier (str): Repository identifier for logging

    Returns:
        Dict: The scan results from Claude
    """
    logging.info(f"Running Claude for staged changes in: {identifier}")

    # Create the prompt for analyzing staged changes
    prompt = f"""Please review the git staged changes in this repository for potential bugs.

Focus on identifying issues that represent objectively incorrect behavior, could lead to exceptions or program crashes, or constitute security vulnerabilities.

Please respond with a JSON object in the following format:

{{
  "issues": [
    {{
      "file": "src/App.tsx",
      "line": 42,
      "description": "Memory leak in useEffect cleanup"
    }}
  ]
}}

If no issues are found, return {{"issues": []}}.
"""

    logging.info(f"Sending prompt to Claude for staged changes analysis")

    try:
        # Call Claude via subprocess
        response_data = run_claude_subprocess(repo_path, prompt)

        # Create the final scan results
        scan_results = {
            "identifier": identifier,
            "issues": response_data,
        }

        return scan_results

    except Exception as e:
        logging.error(f"Error processing staged changes: {e}")
        raise


def process_entry(
    url: str,
    commit_sha: str,
    work_dir: Path,
    results_dir: Path,
    resume: bool = False,
) -> Tuple[bool, str]:
    """
    Process a single PR or commit URL with specified commit SHA.

    If resume=True and results already exist for this entry, they will be loaded
    from disk instead of reprocessing the entry.
    """
    logging.info(f"Processing URL: {url} with commit SHA: {commit_sha}")

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

    # Create identifier based on commit SHA
    identifier = f"{owner}_{repo}_{commit_sha[:7]}"

    # Check if we already have results for this entry
    result_file = results_dir / f"claude_scan_{identifier}.json"
    if resume and result_file.exists():
        logging.info(
            f"Found existing results for {url} with SHA {commit_sha}, loading from disk"
        )
        try:
            with open(result_file, "r") as f:
                scan_res = json.load(f)
            return True, f"Loaded existing results for {url} with SHA {commit_sha}"
        except Exception as e:
            logging.warning(
                f"Failed to load existing results for {url} with SHA {commit_sha}, re-scanning",
                exc_info=True,
            )

    # Clone the repository
    local_repo = work_dir / identifier
    if local_repo.exists():
        logging.info(f"Removing existing repository at {local_repo}")
        shutil.rmtree(local_repo)

    logging.info(f"Cloning repository {repo_url} to {local_repo}")
    try:
        repo = Repo.clone_from(repo_url, local_repo)
    except Exception as e:
        logging.error(f"Failed to clone repository: {e}")
        return False, f"Failed to clone repository: {str(e)}"

    try:
        # Checkout the specified commit SHA
        logging.info(f"Checking out commit SHA {commit_sha}")
        repo.git.checkout(commit_sha)

        # Reset to the previous commit and stage all changes
        logging.info("Running git reset HEAD~1 to go back one commit")
        repo.git.reset("HEAD~1")

        logging.info("Running git add -A to stage all changes")
        repo.git.add("-A")

        # Check if there are any staged changes
        if not repo.git.diff("--cached"):
            logging.warning("No staged changes found after git operations")
            return False, f"No staged changes found for commit {commit_sha}"

        # Run Claude on the staged changes
        logging.info(f"Running Claude analysis on staged changes")
        scan_res = run_claude_on_staged_changes(local_repo, identifier)

        # Save the results
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w") as f:
            json.dump(scan_res, f, indent=2)

        logging.info(f"Saved results to {result_file}")

        issues_count = len(scan_res.get("issues", []))
        return (
            True,
            f"Processed {url} with SHA {commit_sha}, found {issues_count} issues",
        )

    except Exception as e:
        logging.error(f"Error during git operations or Claude analysis: {e}")
        return False, f"Error processing {url} with SHA {commit_sha}: {str(e)}"
    finally:
        # Clean up the cloned repository
        if local_repo.exists():
            logging.debug(f"Cleaning up repository at {local_repo}")
            shutil.rmtree(local_repo)


###############################################################################
# CLI                                                                         #
###############################################################################


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


def main(argv: List[str] | None = None) -> None:
    argv = argv or sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Bug-fix proximity evaluation using Claude CLI on staged changes."
    )
    parser.add_argument(
        "csv_file", type=Path, help="CSV file with PR/commit URLs and commit SHAs."
    )
    parser.add_argument("--tmpdir", type=Path, help="Directory to store clones.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("claude_pr_results"),
        help="Directory to store scan results (default: claude_pr_results)",
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
        help="Randomly shuffle the entries before processing",
    )
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

    # Shuffle entries if requested
    if args.shuffle:
        logging.info(f"Shuffling {len(entries)} entries")
        random.shuffle(entries)

    if args.tmpdir:
        work_dir = args.tmpdir
        logging.info(f"Using specified temporary directory: {work_dir}")
        work_dir.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory(prefix="claude-pr-eval-")
        work_dir = Path(tmp.name)
        logging.info(f"Created temporary directory: {work_dir}")
        cleanup = True

    # Create results directory
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving scan results to {results_dir}")

    # Function to process a single entry for parallel execution
    def process_entry_wrapper(entry_index_tuple):
        index, (url, commit_sha) = entry_index_tuple
        logging.info(
            f"Processing entry {index+1}/{len(entries)}: {url} with SHA {commit_sha}"
        )
        try:
            # Create a unique temporary directory for this entry
            entry_work_dir = work_dir / f"entry_{index}"
            entry_work_dir.mkdir(parents=True, exist_ok=True)

            ok, msg = process_entry(
                url, commit_sha, entry_work_dir, results_dir, args.resume
            )

            # Clean up the temporary directory
            if entry_work_dir.exists():
                shutil.rmtree(entry_work_dir)

            return ok, msg, url, commit_sha, None  # None means no exception
        except KeyboardInterrupt:
            return False, "Interrupted", url, commit_sha, "KeyboardInterrupt"
        except Exception as e:
            logging.exception(f"Error processing {url} with SHA {commit_sha}")
            return False, f"Error: {str(e)}", url, commit_sha, str(e)

    successes = 0

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
                    i, (url, commit_sha) = future_to_entry[future]
                    try:
                        ok, msg, url, commit_sha, exception = future.result()
                        print(msg)
                        if ok:
                            successes += 1
                            logging.info(
                                f"Successfully processed {url} with SHA {commit_sha}"
                            )
                        else:
                            if exception:
                                logging.error(
                                    f"Error processing {url} with SHA {commit_sha}: {exception}"
                                )
                            else:
                                logging.info(
                                    f"Failed to process {url} with SHA {commit_sha}"
                                )
                    except Exception as e:
                        logging.exception(
                            f"Error getting result for {url} with SHA {commit_sha}"
                        )

                    # Update progress bar
                    progress_bar.update(1)
    else:
        # Sequential processing (original behavior)
        for i, (url, commit_sha) in enumerate(tqdm(entries, desc="Processing")):
            logging.info(
                f"Processing entry {i+1}/{len(entries)}: {url} with SHA {commit_sha}"
            )
            try:
                # Create a unique temporary directory for this entry
                entry_work_dir = work_dir / f"entry_{i}"
                entry_work_dir.mkdir(parents=True, exist_ok=True)

                ok, msg = process_entry(
                    url, commit_sha, entry_work_dir, results_dir, args.resume
                )

                # Clean up the temporary directory
                if entry_work_dir.exists():
                    shutil.rmtree(entry_work_dir)

                print(msg)
                if ok:
                    successes += 1
                    logging.info(f"Successfully processed {url} with SHA {commit_sha}")
                else:
                    logging.info(f"Failed to process {url} with SHA {commit_sha}")
            except KeyboardInterrupt:
                break
            except Exception as e:
                logging.exception("Error processing %s with SHA %s", url, commit_sha)
                print(f"Error processing {url} with SHA {commit_sha}: {str(e)}")

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
