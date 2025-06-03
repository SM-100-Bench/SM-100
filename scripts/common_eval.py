import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Optional, Set, Any, Tuple

import requests
from unidiff import PatchSet  # type: ignore

from bismuthsdk import V1ScanResult

###############################################################################
# Environment / tokens
###############################################################################

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or os.getenv("GITHUB_PAT")
if not GITHUB_TOKEN:
    logging.warning(
        "Environment variable GITHUB_TOKEN (or GITHUB_PAT) is not set – "
        "GitHub rate limits will be very low."
    )

HEADERS = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}" if GITHUB_TOKEN else "",
    "X-GitHub-Api-Version": "2022-11-28",
}

# Directory to cache diffs
DIFF_CACHE_DIR = Path("diff_cache")

###############################################################################
# Lightweight GitHub helpers
###############################################################################

PR_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/pull/(?P<number>\d+)"
)
COMMIT_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/commit/(?P<sha>[0-9a-fA-F]{7,40})"
)
REPO_URL_RE = re.compile(r"https?://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)")

###############################################################################
# Core helpers
###############################################################################


def github_get(url: str, params: dict | None = None) -> dict:
    """Light wrapper around requests.get that returns JSON or raises."""
    r = requests.get(url, headers=HEADERS, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"GitHub API error {r.status_code}: {r.text}")
    return r.json()


def get_cached_diff(cache_key: str) -> Optional[str]:
    """
    Check if a diff is cached and return it if it exists.

    Args:
        cache_key: A unique identifier for the diff

    Returns:
        The cached diff text if it exists, None otherwise
    """
    cache_file = DIFF_CACHE_DIR / f"{cache_key}.diff"
    if cache_file.exists():
        logging.info(f"Using cached diff from {cache_file}")
        with open(cache_file, "r") as f:
            return f.read()
    return None


def save_diff_to_cache(cache_key: str, diff_text: str) -> None:
    """
    Save a diff to the cache.

    Args:
        cache_key: A unique identifier for the diff
        diff_text: The diff text to cache
    """
    DIFF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = DIFF_CACHE_DIR / f"{cache_key}.diff"
    with open(cache_file, "w") as f:
        f.write(diff_text)
    logging.info(f"Saved diff to cache: {cache_file}")


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

    # Create a cache key for this PR
    cache_key = f"pr_{owner}_{repo}_{pr_number}"

    # Check if we have a cached diff
    diff_text = get_cached_diff(cache_key)

    if diff_text is None:
        logging.info(f"Fetching PR files and building line map")
        for i in range(5):
            try:
                diff_req = requests.get(pr_info["diff_url"], headers=HEADERS)
                diff_req.raise_for_status()
                diff_text = diff_req.text
                # Save the diff to cache
                save_diff_to_cache(cache_key, diff_text)
                break
            except Exception as e:
                logging.info(f"Exception getting diff {e}, retrying")
                if i < 4:  # Don't sleep after the last attempt
                    time.sleep([10, 30, 60, 180, 300][i])
                else:
                    logging.error(f"Failed to fetch diff after 5 retries")
                    raise

    _merge_line_maps(gold, _parse_patch_line_map(diff_text))

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

    # Create a cache key for this commit
    cache_key = f"commit_{owner}_{repo}_{sha}"

    # Check if we have a cached diff
    diff_text = get_cached_diff(cache_key)

    if diff_text is None:
        logging.info(f"Fetching commit diff for {owner}/{repo} commit {sha}")

        for i in range(5):
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
                break
            except Exception as e:
                logging.info(f"Exception getting diff {e}, retrying")
                if i < 4:  # Don't sleep after the last attempt
                    time.sleep([10, 30, 60, 180, 300][i])
                else:
                    logging.error(f"Failed to fetch diff after 5 retries")
                    raise

    _merge_line_maps(gold, _parse_patch_line_map(diff_text))

    logging.info(f"Processed files from commit, found changes in {len(gold)} files")
    return parent_sha, gold


# ------------------------------------------------------------------ diff utils
def _parse_patch_line_map(patch_text: str) -> Dict[str, Set[int]]:
    """
    Convert a unified diff **patch** (as text) into a mapping
    {file_path -> {line_numbers_modified_in_target_file}}.

    Added *and* modified lines are included.  For removals we fall back to the
    source line numbers so that deletions can still match later reports.
    """
    line_map: Dict[str, Set[int]] = {}
    if not patch_text:
        return line_map

    try:
        ps = PatchSet(patch_text.splitlines())
    except Exception:
        # Gracefully handle malformed patches
        return line_map

    for f in ps:
        nums: Set[int] = set()
        for hunk in f:
            for ln in hunk:
                if ln.is_added and ln.target_line_no:
                    nums.add(ln.target_line_no)
                elif ln.is_removed and ln.source_line_no:
                    nums.add(ln.source_line_no)
        if nums:
            line_map[f.path] = nums

    return line_map


def _merge_line_maps(target: Dict[str, Set[int]], src: Dict[str, Set[int]]) -> None:
    """Merge *src* into *target* (union of line-number sets)."""
    for fp, lines in src.items():
        target.setdefault(fp, set()).update(lines)


# ------------------------------------------------------------------ results logging
def get_result_filename(
    identifier: str, results_dir: Path, prefix: str = "scan"
) -> Path:
    """
    Generate a consistent filename for results based on identifier.

    Args:
        identifier: A unique identifier (e.g., URL or commit SHA)
        results_dir: Directory to save results in
        prefix: Prefix for the filename

    Returns:
        Path to the result file (may not exist yet)
    """
    # Create a safe filename from the identifier
    safe_id = re.sub(r"[^\w\-]", "_", identifier)
    if len(safe_id) > 100:  # Avoid excessively long filenames
        safe_id = safe_id[:100]

    return results_dir / f"{prefix}_{safe_id}.json"


def result_exists(identifier: str, results_dir: Path, prefix: str = "scan") -> bool:
    """
    Check if results already exist for the given identifier.

    Args:
        identifier: A unique identifier (e.g., URL or commit SHA)
        results_dir: Directory to check for results
        prefix: Prefix used for the filename

    Returns:
        True if results exist, False otherwise
    """
    result_file = get_result_filename(identifier, results_dir, prefix)
    return result_file.exists()


def load_scan_results(
    identifier: str, results_dir: Path, prefix: str = "scan"
) -> V1ScanResult:
    """
    Load existing scan results from a file.

    Args:
        identifier: A unique identifier (e.g., URL or commit SHA)
        results_dir: Directory to load results from
        prefix: Prefix used for the filename

    Returns:
        Dictionary containing the loaded results

    Raises:
        FileNotFoundError: If the results file doesn't exist
    """
    result_file = get_result_filename(identifier, results_dir, prefix)
    if not result_file.exists():
        raise FileNotFoundError(f"No results found for {identifier}")

    with open(result_file, "r") as f:
        return V1ScanResult.model_validate_json(f.read())


def save_scan_results(
    res: V1ScanResult, identifier: str, results_dir: Path, prefix: str = "scan"
) -> Path:
    """
    Save scan results to a JSON file in the results directory.

    Args:
        results: The scan results object to save
        identifier: A unique identifier for this scan (e.g., PR URL or commit SHA)
        results_dir: Directory to save results in
        prefix: Prefix for the filename

    Returns:
        Path to the saved file
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    result_file = get_result_filename(identifier, results_dir, prefix)

    with open(result_file, "w") as f:
        f.write(res.model_dump_json())

    logging.info(f"Saved scan results to {result_file}")
    return result_file


# Export all public functions and constants
__all__ = [
    "HEADERS",
    "DIFF_CACHE_DIR",
    "github_get",
    "get_cached_diff",
    "save_diff_to_cache",
    "get_pr_base_and_line_map",
    "get_commit_parent_and_line_map",
    "_parse_patch_line_map",
    "_merge_line_maps",
    "save_scan_results",
    "load_scan_results",
    "result_exists",
    "get_result_filename",
    "PR_URL_RE",
    "COMMIT_URL_RE",
    "REPO_URL_RE",
]
