import argparse
import concurrent.futures
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
from typing import Dict, Iterable, List, Optional, Set, Tuple

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


###############################################################################
# Claude CLI integration                                                      #
###############################################################################


def clone_repo_at_commit(repo_url: str, commit_sha: str, clone_dir: Path) -> Path:
    """
    Clone a repository at a specific commit.

    Args:
        repo_url (str): The URL of the repository to clone
        commit_sha (str): The commit SHA to checkout
        clone_dir (Path): The directory to clone into

    Returns:
        Path: The path to the cloned repository
    """
    logging.info(f"Cloning repository {repo_url} at commit {commit_sha}")

    # Create a unique directory name for this repo/commit
    repo_name = repo_url.split("/")[-1]
    repo_dir = clone_dir / f"{repo_name}_{commit_sha[:7]}"

    # Remove the directory if it already exists
    if repo_dir.exists():
        logging.info(f"Removing existing directory {repo_dir}")
        shutil.rmtree(repo_dir)

    # Clone the repository
    logging.info(f"Cloning into {repo_dir}")
    repo = Repo.clone_from(f"https://github.com/{repo_url}.git", repo_dir)

    # Checkout the specific commit
    logging.info(f"Checking out commit {commit_sha}")
    repo.git.checkout(commit_sha)

    return repo_dir


def run_claude_subprocess(prompt: str, cwd: Path) -> Dict:
    """
    Run the claude CLI program as a subprocess with the given prompt.

    Args:
        prompt (str): The prompt to send to Claude
        cwd (Optional[Path]): The directory to run the command from

    Returns:
        Dict: The parsed JSON response from Claude
    """
    try:
        # Run the claude command with the prompt and JSON output format
        cmd = ["claude", "-p", prompt, "--output-format", "json"]

        logging.info(
            f"Running claude command: {' '.join(cmd[:2])} '<prompt>' {' '.join(cmd[3:])}"
        )

        logging.info(f"Running from directory: {cwd}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=900,  # 15 minute timeout
            cwd=str(cwd),
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
# Core processing                                                             #
###############################################################################


def run_claude(repo_id, subsystems, repo_dir):
    """
    Run Claude on a repository with specified subsystems.

    Args:
        repo_id (str): Repository ID in the format {repo_owner}_{repo_name}_{pr_number_or_commit}
        subsystems (list): List of subsystems, each a dict with "name" and "files" keys
        repo_dir (Path): Path to the cloned repository

    Returns:
        object: The scan results from Claude
    """
    logging.info(f"Running Claude for repo ID: {repo_id} in directory {repo_dir}")

    all_issues = []

    # Process each subsystem sequentially
    for i, subsystem in enumerate(subsystems):
        subsystem_name = subsystem["name"]
        files = ", ".join([f"'{file}'" for file in subsystem["files"]])

        logging.info(f"Processing subsystem {i+1}/{len(subsystems)}: {subsystem_name}")

        # Create the prompt for this subsystem
        prompt = f"""Please review the code in the {subsystem_name} subsystem (consisting of {files}) in the current directory for potential bugs.

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

        logging.info(f"Sending prompt to Claude for subsystem {subsystem_name}")

        try:
            # Call Claude via subprocess from within the repository directory
            response_data = run_claude_subprocess(prompt, repo_dir)

            all_issues.append(response_data)

        except Exception as e:
            logging.error(f"Error processing subsystem {subsystem_name}: {e}")
            # Continue with other subsystems even if one fails
            continue

    # Create the final scan results
    scan_results = {
        "repo_id": repo_id,
        "issues": all_issues,
    }

    logging.info(f"Total issues found across all subsystems: {len(all_issues)}")
    return scan_results


def process_url(
    url: str,
    results_dir: Path,
    tmp_dir: Path,
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

    # Get the base/parent commit and line map
    if pr_match:
        logging.info(f"Processing as PR #{pr_match['number']}")
        base_sha, head_sha, gold_map = get_pr_base_and_line_map(
            owner, repo, int(pr_match["number"])
        )
        commit_to_checkout = base_sha  # For PRs, we want to check out the base commit
    else:
        sha = commit_match["sha"]
        logging.info(f"Processing as commit {sha}")
        parent_sha, gold_map = get_commit_parent_and_line_map(owner, repo, sha)
        commit_to_checkout = (
            parent_sha  # For commits, we want to check out the parent commit
        )

    if not gold_map:
        logging.error(f"No changes found in {url}")
        return False, f"No changes found in {url}"

    scan_res = None
    if resume and result_exists(url, results_dir, prefix="claude_scan"):
        logging.info(f"Found existing results for {url}, attempting to load")
        try:
            with open(
                get_result_filename(url, results_dir, prefix="claude_scan"), "r"
            ) as f:
                scan_res = json.load(f)
        except:
            logging.warning(
                f"Failed to load existing results for {url}, re-scanning", exc_info=True
            )

    if scan_res is None:
        logging.info(f"Running Claude scan on repository")

        # Determine repo_id based on the URL
        if pr_match:
            repo_id = f"{owner}_{repo}_{pr_match['number']}"
        else:
            repo_id = f"{owner}_{repo}_{commit_match['sha']}"

        logging.info(f"Using repo ID: {repo_id}")

        # Load subsystems from JSON file
        subsystems_file = Path(f"subsystems/{repo_id}.json")

        logging.info(f"Loading subsystems from {subsystems_file}")

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

        # Clone the repository at the base/parent commit
        repo_dir = clone_repo_at_commit(
            repo_full, commit_to_checkout, tmp_dir
        )

        try:
            # Run Claude with the repo_id, filtered subsystems, and repo directory
            scan_res = run_claude(repo_id, subsystems, repo_dir)
        finally:
            # Clean up the cloned repository
            if repo_dir.exists():
                logging.info(f"Cleaning up repository directory: {repo_dir}")
                shutil.rmtree(repo_dir)

    # Save both the scan results and our analysis
    logging.info(f"Saving scan results to {results_dir}")
    with open(get_result_filename(url, results_dir, prefix="claude_scan"), "w") as f:
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
    parser = argparse.ArgumentParser(
        description="Bug-fix proximity evaluation using Claude CLI."
    )
    parser.add_argument("url_file", type=Path, help="File with PR/commit URLs.")
    parser.add_argument(
        "--tmpdir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "claude_eval_repos",
        help="Directory to store temporary repository clones.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("claude_results"),
        help="Directory to store scan results (default: claude_results)",
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

    # Create temporary directory for clones if it doesn't exist
    tmp_dir = args.tmpdir
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Using temporary directory for clones: {tmp_dir}")

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
            # Create a unique temporary directory for this URL
            url_tmp_dir = tmp_dir / f"url_{index}"
            url_tmp_dir.mkdir(parents=True, exist_ok=True)

            ok, msg = process_url(url, results_dir, url_tmp_dir, args.resume)

            # Clean up the temporary directory
            if url_tmp_dir.exists():
                shutil.rmtree(url_tmp_dir)

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
                # Create a unique temporary directory for this URL
                url_tmp_dir = tmp_dir / f"url_{i}"
                url_tmp_dir.mkdir(parents=True, exist_ok=True)

                ok, msg = process_url(u, results_dir, url_tmp_dir, args.resume)

                # Clean up the temporary directory
                if url_tmp_dir.exists():
                    shutil.rmtree(url_tmp_dir)

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
