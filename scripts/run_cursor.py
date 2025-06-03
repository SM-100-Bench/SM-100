#!/usr/bin/env python3
import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import subprocess
import requests
from selenium import webdriver
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from git import Repo  # type: ignore
from tqdm import tqdm  # progress bar

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    github_get,
    HEADERS,
    get_result_filename,
    result_exists,
    _parse_patch_line_map,
    _merge_line_maps,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def drive_cursor(electron_app_path: str, repo: Path, subsystems: list) -> list:
    """
    Args:
        electron_app_path (str): Path to the Electron app executable
        repo (Path): Path to the repository to analyze
        subsystems (list): List of subsystems, each a dict with "name" and "files" keys

    Returns:
        list: Content of the code blocks
    """
    # Setup for launching Cursor and connecting to it
    data_dir = Path("~/Library/Application Support/Cursor").expanduser()

    # Launch Cursor with the repo as a positional argument
    logging.info(f"Launching Cursor with repo: {repo.absolute()}")
    cursor_proc = subprocess.Popen(
        [
            electron_app_path,
            "--remote-debugging-port=9222",
            f"--user-data-dir={data_dir}",
            str(repo.absolute()),
        ]
    )

    # Give Cursor a moment to start up
    time.sleep(2)

    # Connect to the debugging port
    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "localhost:9222")

    # Start the Electron app
    service = Service(ChromeDriverManager(driver_version="132.0.6834.0").install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    logging.info("Chrome driver initialized with remote debugging connection")

    all_results = []

    try:
        # Wait for app to fully load
        time.sleep(5)
        logging.info("Initial wait for app to load completed")

        # Open Cursor settings
        actions = ActionChains(driver)
        actions.key_down(Keys.SHIFT).key_down(Keys.COMMAND).send_keys("j").key_up(
            Keys.COMMAND
        ).key_up(Keys.SHIFT).perform()
        logging.info("Pressed Shift+Cmd+J to open Cursor settings")

        time.sleep(1)
        logging.info("Waited 1 second after opening settings")

        # Click on Features menu item
        features_menu_item = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (
                    By.XPATH,
                    "//div[contains(@class, 'settings-menu-item') and descendant::text()='Features']",
                )
            )
        )
        features_menu_item.click()
        logging.info("Clicked on 'Features' menu item")

        # Wait codebase indexing to finish
        WebDriverWait(driver, 300).until(
            EC.presence_of_element_located(
                (
                    By.XPATH,
                    "//div[contains(@class, 'cursor-button') and text()='Resync Index']",
                )
            )
        )
        logging.info("Found 'Resync Index' button, indexing has finished")

        # Process each subsystem
        for i, subsystem in enumerate(subsystems):
            subsystem_name = subsystem["name"]
            files = ", ".join([f"'{file}'" for file in subsystem["files"]])

            logging.info(
                f"Processing subsystem {i+1}/{len(subsystems)}: {subsystem_name}"
            )

            # New chat for each subsystem
            add_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "a.action-label.codicon.codicon-add-two")
                )
            )
            add_button.click()
            logging.info(f"Started a new chat for subsystem {subsystem_name}")

            # Wait for and click on the editor input
            editor_input = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.CLASS_NAME, "aislash-editor-input"))
            )
            editor_input.click()

            message = f"""Examine the '{subsystem_name}' subsystem ({files}) for bugs. Look for objectively wrong behavior, bugs that might cause exceptions, and security vulnerabilities. Be thorough and examine every file completely. Do not attempt to write or suggest fixes, only examine and report the bugs. Report issues found in a JSON markdown block, with each issue in the form {{
      "file": "src/App.tsx",
      "line": 42,
      "description": "Memory leak in useEffect cleanup"
    }}"""

            # Create ActionChains instance for complex key combinations
            actions = ActionChains(driver)

            # Type the message with Shift+Enter for newlines
            # Split the message by newline and then join with Shift+Enter actions
            message_parts = message.split("\n")

            # Type the first part without a preceding newline
            editor_input.send_keys(message_parts[0])

            # For each subsequent part, send Shift+Enter and then the text
            for j, part in enumerate(message_parts[1:], 1):
                # Send Shift+Enter
                actions.key_down(Keys.SHIFT).send_keys(Keys.RETURN).key_up(
                    Keys.SHIFT
                ).perform()
                time.sleep(0.1)  # Small delay to ensure the keypress is registered

                # Type the next part of the message
                editor_input.send_keys(part)

            # Press Enter
            editor_input.send_keys(Keys.RETURN)
            logging.info(f"Sent message for subsystem {subsystem_name}")

            # Wait for generating thing at bottom to appear
            WebDriverWait(driver, 30).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div[forcererender="generating"]')
                )
            )
            logging.info("Generation indicator appeared")

            # and disappear
            WebDriverWait(driver, 120).until_not(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, 'div[forcererender="generating"]')
                )
            )
            logging.info(f"Generation completed for subsystem {subsystem_name}")

            # Retrieve the code blocks
            code_blocks = driver.find_elements(
                By.CSS_SELECTOR, ".composer-message-codeblock"
            )
            subsystem_results = [
                block.get_attribute("innerText") for block in code_blocks
            ]
            logging.info(
                f"Retrieved {len(subsystem_results)} code blocks for subsystem {subsystem_name}"
            )

            # Add to overall results
            all_results.extend(subsystem_results)

        return all_results

    finally:
        driver.quit()
        cursor_proc.terminate()
        logging.info("Browser closed")


###############################################################################
# GitHub diff acquisition                                                     #
###############################################################################


def get_pr_base_and_line_map(
    owner: str, repo: str, pr_number: int
) -> Tuple[str, str, Dict[str, Set[int]]]:
    """
    Return (base_sha, head_sha, gold_line_map) for the PR.
    """
    logging.info(f"Fetching PR information for {owner}/{repo}#{pr_number}")
    api_base = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_number}"
    pr_info = github_get(api_base)

    base_sha = pr_info["base"]["sha"]
    head_sha = pr_info["head"]["sha"]
    logging.debug(f"PR base SHA: {base_sha}, head SHA: {head_sha}")

    gold: Dict[str, Set[int]] = {}

    logging.info(f"Fetching PR files and building line map")
    for i in range(5):
        try:
            diff_req = requests.get(pr_info["diff_url"], headers=HEADERS)
            diff_req.raise_for_status()
            break
        except Exception as e:
            logging.info(f"Exception getting diff {e}, retrying")
            time.sleep([10, 30, 60, 180, 300][i])
    _merge_line_maps(gold, _parse_patch_line_map(diff_req.text))

    logging.info(f"Found changes in {len(gold)} files")
    return base_sha, head_sha, gold


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

    gold: Dict[str, Set[int]] = {}

    logging.info(f"Fetching commit diff for {owner}/{repo} commit {sha}")

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

        _merge_line_maps(gold, _parse_patch_line_map(response.text))
    except Exception as e:
        raise Exception("Failed to fetch commit diff")

    logging.info(f"Processed files from commit, found changes in {len(gold)} files")
    return parent_sha, gold


###############################################################################
# Core processing                                                             #
###############################################################################


def process_url(
    url: str,
    work_dir: Path,
    results_dir: Path,
    electron_app_path: str,
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
    repo_url = f"https://github.com/{repo_full}.git"
    logging.info(f"Identified repository: {repo_full}")

    if pr_match:
        pr_number = pr_match["number"]
        identifier = f"{owner}_{repo}_{pr_number}"
    else:
        commit_sha = commit_match["sha"]
        identifier = f"{owner}_{repo}_{commit_sha[:7]}"

    # Check if we already have results for this URL
    result_file = results_dir / f"{identifier}.json"
    if resume and result_file.exists():
        logging.info(f"Found existing results for {url}, skipping")
        return True, f"Skipped {url} (results already exist)"

    # Get the gold_map (modified files and lines)
    if pr_match:
        pr_number = pr_match["number"]
        identifier = f"{owner}_{repo}_{pr_number}"
        logging.info(f"Processing as PR #{pr_number}")
        base_sha, head_sha, gold_map = get_pr_base_and_line_map(
            owner, repo, int(pr_number)
        )
        checkout_sha = base_sha
    else:
        commit_sha = commit_match["sha"]
        identifier = f"{owner}_{repo}_{commit_sha}"
        logging.info(f"Processing as commit {commit_sha}")
        parent_sha, gold_map = get_commit_parent_and_line_map(owner, repo, commit_sha)
        checkout_sha = parent_sha

    if not gold_map:
        logging.error(f"No changes found in {url}")
        return False, f"No changes found in {url}"

    # Clone the repository
    local_repo = work_dir / identifier
    if local_repo.exists():
        logging.info(f"Removing existing repository at {local_repo}")
        shutil.rmtree(local_repo)

    logging.info(f"Cloning repository {repo_url} to {local_repo}")
    repo = Repo.clone_from(repo_url, local_repo)
    logging.info(f"Checking out SHA {checkout_sha}")
    repo.git.checkout(checkout_sha)

    # Load subsystems from JSON file
    subsystems_file = Path(f"subsystems/{identifier}.json")
    if not subsystems_file.exists():
        logging.warning(f"Subsystems file not found: {subsystems_file}")
        raise FileNotFoundError(f"Subsystems file not found: {subsystems_file}")

    logging.info(f"Loading subsystems from {subsystems_file}")

    with open(subsystems_file, "r") as f:
        subsystems = json.load(f)

    # Filter subsystems to only include those with files that overlap with gold_map
    filtered_subsystems = [
        s for s in subsystems if set(s["files"]) & set(gold_map.keys())
    ]

    if not filtered_subsystems:
        logging.warning(
            "No subsystems with overlapping files found"
        )
        return False, "No subsystems with overlapping files found"

    if sum(sum(len(f) for f in subsystem["files"]) for subsystem in filtered_subsystems) > 5000:
        return False, "Subsystem file list too long"

    # Run Cursor on the repository
    logging.info(f"Running Cursor on repository {local_repo}")
    code_blocks = drive_cursor(electron_app_path, local_repo, filtered_subsystems)

    # Save the results
    results = {
        "url": url,
        "identifier": identifier,
        "code_blocks": code_blocks,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Saved results to {result_file}")

    return True, f"Processed {url}, found {len(code_blocks)} code blocks"


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
        description="Cursor evaluation on GitHub repositories."
    )
    parser.add_argument("url_file", type=Path, help="File with PR/commit URLs.")
    parser.add_argument("--tmpdir", type=Path, help="Directory to store clones.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("cursor_results"),
        help="Directory to store scan results (default: cursor_results)",
    )
    parser.add_argument(
        "--cursor-path",
        type=str,
        default="/Applications/Cursor.app/Contents/MacOS/Cursor",
        help="Path to the Cursor executable (default: /Applications/Cursor.app/Contents/MacOS/Cursor)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skipping already processed URLs",
    )
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
        tmp = tempfile.TemporaryDirectory(prefix="cursor-eval-")
        work_dir = Path(tmp.name)
        logging.info(f"Created temporary directory: {work_dir}")
        cleanup = True

    # Create results directory
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving results to {results_dir}")

    successes = 0
    for i, u in enumerate(tqdm(urls, desc="Processing")):
        logging.info(f"Processing URL {i+1}/{len(urls)}: {u}")
        try:
            ok, msg = process_url(
                u, work_dir, results_dir, args.cursor_path, args.resume
            )
            print(msg)
            if ok:
                successes += 1
                logging.info(f"Successfully processed {u}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            logging.exception("Error processing %s", u)
            print(f"Error processing {u}: {str(e)}")

    print("\n==== SUMMARY ====")
    print(f"Processed : {len(urls)}")
    print(f"Successful: {successes}")
    logging.info(
        f"Final results: processed {len(urls)} URLs, successfully processed {successes}"
    )

    if cleanup:
        logging.debug(f"Cleaning up temporary directory: {work_dir}")
        tmp.cleanup()


if __name__ == "__main__":
    main()
