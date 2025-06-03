#!/usr/bin/env python3
"""
Script to construct review prompts and save them to files.
Usage: python construct_prompts.py output_directory
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

from common_eval import (
    PR_URL_RE,
    COMMIT_URL_RE,
    get_pr_base_and_line_map,
    get_commit_parent_and_line_map,
    DIFF_CACHE_DIR,
)


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


def get_gold_map(
    owner: str, repo: str, identifier: str, is_pr: bool
) -> Dict[str, Set[int]]:
    """
    Get the gold map (files changed in PR/commit) for a repository.

    Args:
        owner: Repository owner
        repo: Repository name
        identifier: PR number or commit SHA
        is_pr: True if identifier is a PR number, False if it's a commit SHA

    Returns:
        Dictionary mapping file paths to sets of line numbers that were modified
    """
    if is_pr:
        _, _, gold_map = get_pr_base_and_line_map(owner, repo, int(identifier))
    else:
        _, gold_map = get_commit_parent_and_line_map(owner, repo, identifier)

    return gold_map


def construct_prompt(repo_id: str, subsystems: List[Dict]) -> str:
    """
    Construct a prompt for the repo.

    Args:
        repo_id: Repository ID in the format {repo_owner}_{repo_name}_{pr_number_or_commit}
        subsystems: List of subsystems, each a dict with "name" and "files" keys

    Returns:
        The constructed prompt
    """
    # Build the subsystems description
    subsystems_description = []
    for i, subsystem in enumerate(subsystems, 1):
        subsystem_name = subsystem["name"]
        files = ", ".join([f"'{file}'" for file in subsystem["files"]])
        subsystems_description.append(
            f"{i}. {subsystem_name} subsystem (consisting of {files})"
        )

    # Create the comprehensive prompt
    prompt = f"""Please review the code in the {repo_id} repository for potential bugs.
Focus on identifying issues that represent objectively incorrect behavior, could lead to exceptions or program crashes, or constitute security vulnerabilities.

Please analyze the following subsystems:

{'\n'.join(subsystems_description)}

Report your findings in the following structured format:

{{
  "issues": [
    {{
      "file": "src/App.tsx",
      "line": 42,
      "description": "Memory leak in useEffect cleanup"
    }}
  ]
}}"""

    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Construct prompts and save them to files."
    )
    parser.add_argument("output_dir", type=Path, help="Directory to save prompt files")
    parser.add_argument(
        "--subsystems-dir",
        type=Path,
        default=Path("subsystems"),
        help="Directory containing subsystem JSON files (default: subsystems)",
    )
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

    # Check if subsystems directory exists
    if not args.subsystems_dir.exists():
        logging.error(f"Subsystems directory '{args.subsystems_dir}' does not exist.")
        sys.exit(1)

    # Find all subsystem JSON files
    subsystem_files = list(args.subsystems_dir.glob("*.json"))

    if not subsystem_files:
        logging.warning(f"No subsystem JSON files found in '{args.subsystems_dir}'.")
        sys.exit(1)

    logging.info(f"Found {len(subsystem_files)} subsystem files.")

    # Process each subsystem file
    for subsystem_file in subsystem_files:
        repo_id = subsystem_file.stem  # Get filename without extension

        try:
            # Parse the repo_id to get owner, repo, and identifier
            owner, repo, identifier, is_pr = parse_repo_id(repo_id)
            logging.info(
                f"Processing {repo_id} (owner={owner}, repo={repo}, identifier={identifier}, is_pr={is_pr})"
            )

            # Get the gold map (files changed in PR/commit)
            gold_map = get_gold_map(owner, repo, identifier, is_pr)
            logging.info(f"Found changes in {len(gold_map)} files")

            # Load subsystems from JSON file
            with open(subsystem_file, "r") as f:
                subsystems = json.load(f)

            # Filter subsystems to only include those with files that overlap with gold_map
            original_count = len(subsystems)
            subsystems = [
                s for s in subsystems if set(s["files"]) & set(gold_map.keys())
            ]

            if not subsystems:
                logging.warning(
                    f"No subsystems with overlapping files found for {repo_id}, using all subsystems"
                )
                # Reload all subsystems
                with open(subsystem_file, "r") as f:
                    subsystems = json.load(f)
            else:
                logging.info(
                    f"Filtered subsystems from {original_count} to {len(subsystems)} based on gold map overlap"
                )

            # Construct the prompt
            prompt = construct_prompt(repo_id, subsystems)

            # Generate output filename
            output_filename = get_safe_filename(repo_id)
            output_path = args.output_dir / output_filename

            # Save prompt to file
            with open(output_path, "w") as f:
                f.write(prompt)

            logging.info(f"Created prompt for {repo_id} -> {output_path}")

        except Exception as e:
            logging.error(f"Error processing {subsystem_file}: {str(e)}", exc_info=True)

    logging.info(f"All prompts have been saved to {args.output_dir}")


if __name__ == "__main__":
    main()
