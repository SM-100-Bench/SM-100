## Setup Scripts
* `construct_prompts.py`: create the results in `scan_prompts/` which are what are passed to interactive systems for needle in haystack bug scanning.
* `construct_pr_prompts.py`: create the results in `pr_review_prompts/` - what is passed to interactive systems for PR review bug scanning.
* `gh_*.py`: manage the mirrored repositories in the SM-100-Bench GitHub organization.

## Precomputes
* `subsystems/`: for each repo, the automatically determined list of subsystems which are used to group and limit the files passed for review.
* `diff_cache/`: github severely ratelimits PR diff fetching, so those are cached here.
* `100.txt`: simple listing of the SM-100 bugs.
* `prs.csv`: CSV of bug to introduction commit, used for PR review tasks.

## Agent Run Scripts
For local desktop agents, `run_*.py` scripts drive the tools and save results.
* Claude Code is driven via the "sdk" (headless mode)
* Cursor is driven via Selenium

## Agent Evaluation Scripts
As the names suggest. Notes:
* `eval_codex_json.py`: evaluates the manually copied output JSON from running Codex - does not drive Codex itself.

## Test Running
We've created environments to automatically test fixes for as many of the SM-100 bugs as possible. Unfortunately many projects do not have robust test frameworks in place, so there are not tests for each and every bug in the dataset. Any fixes for those have to be manually validated.
* `test_eval/`: dir holding per-bug fixtures for testing. Dockerfiles, inner test scripts, basic checkers for simple text presence/absense.
* `eval_tests.py`: run tests given patches from the agents.

## Basic Agent
* The basic agent used as a baseline and for open models, similar to what is described by Anthropic in their basic agent for SWE-Bench. Has the following tools: `bash`, `str_replace_editor` (all in one file + dir viewing, editing), `think`, `report_bug` (to feed out structured bug reports), `finish`.
