import argparse
import concurrent.futures
import json
import logging
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import asyncio
from typing import Any, Iterable, List, Optional, TypedDict, Union
import uuid
import atexit
from asimov.services.inference_clients import (
    InferenceClient,
    OpenRouterInferenceClient,
    ChatMessage,
    ChatRole,
)
from git import Repo
from tqdm import tqdm

from gasp import Parser, Deserializable
from gasp.template_helpers import interpolate_prompt

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


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters (except at start)
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters followed by lowercase
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


class Bash(Deserializable):
    """
    Run commands in a bash shell

    * When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
    * You don't have access to the internet via this tool.
    * You do have access to a mirror of common linux and python packages via apt and pip.
    * State is persistent across command calls and discussions with the user.
    * To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
    * Please avoid commands that may produce a very large amount of output.
    * Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.

    Attributes:
        command (str): The bash command to run.
    """

    command: str


class StrReplaceEditor(Deserializable):
    """
    Custom editing tool for viewing, creating and editing files

    * State is persistent across command calls and discussions with the user
    * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
    * The `create` command cannot be used if the specified `path` already exists as a file
    * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
    * The `undo_edit` command will revert the last edit made to the file at `path`

    Notes for using the `str_replace` command:
    * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
    * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
    * The `new_str` parameter should contain the edited lines that should replace the `old_str`

    Attributes:
        command (str): The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.
        path (str): Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.
        file_text (str, optional): Required parameter of `create` command, with the content of the file to be created.
        insert_line (int, optional): Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.
        new_str (str, optional): Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.
        old_str (str, optional): Required parameter of `str_replace` command containing the string in `path` to replace.
        view_range (list[int], optional): Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.
    """

    command: str
    path: str
    file_text: str = None
    insert_line: int = None
    new_str: str = None
    old_str: str = None
    view_range: list[int] = None


class Think(Deserializable):
    """
    Freely describe and reflect on what you know so far, things that you tried, and how that aligns with your objective and the user's intent. You can play through different scenarios, weigh options, and reason about possible next next steps. The user will not see any of your thoughts here, so you can think freely.

    Attributes:
        thoughts (list[str]): Array of thoughts and reflections.
    """

    thoughts: list[str]


class ReportBug(Deserializable):
    """
    Report a bug in the codebase

    * This tool is used to report bugs found during the review process.
    * The `bug_description` parameter should contain a clear and concise description of the bug.
    * The `file_path` parameter should point to the file where the bug was found.
    * The `line_number` parameter should indicate the line number where the bug was found.

    Attributes:
        bug_description (str): A clear and concise description of the bug.
        file_path (str): The path to the file where the bug was found.
        line_number (int): The line number where the bug was found.
    """

    bug_description: str
    file_path: str
    line_number: int


class Finish(Deserializable):
    """
    Stop the review loop. Only use this when you're certain you have explored thoroughly and fixed all found bugs.

    This class has no attributes as it serves as a simple signal to terminate the review process.
    """

    pass


type ToolTypes = Union[
    Bash,
    StrReplaceEditor,
    Think,
    ReportBug,
    Finish,
]


class ToolCall(TypedDict):
    """
    TypedDict for tool call representation.

    Attributes:
        name: Name of the tool to call
        input: Input parameters for the tool
    """

    name: str
    input: ToolTypes


async def tool_parser_func(llm_output: str, mode) -> list[dict]:
    parser = Parser(ToolTypes)
    _result = parser.feed(llm_output)

    # Check if parsing is complete
    if parser.is_complete():
        validated = parser.validate()

        if validated:
            tool_name = camel_to_snake(validated.__class__.__name__)

            return [{"name": tool_name, "input": validated}]

    # Parser is not complete or validation failed
    return []


def tool_result_reducer(tool_calls: list[ToolCall], results: list[Any]) -> str:
    if not results:
        return ""

    # Build formatted output with separators
    formatted_results = []

    for i, (tool_call, result) in enumerate(zip(tool_calls, results)):
        # Create separator with tool name
        separator = f"==== {tool_call['name']}_result ===="
        formatted_results.append(separator)

        if result is None:
            # Handle None results with error message
            error_msg = f"The {tool_call['name']} tool encountered an error during execution and did not complete."
            formatted_results.append(error_msg)
        else:
            formatted_results.append(str(result))

        formatted_results.append("")  # Empty line for spacing

    # Join all parts and strip trailing newline
    return "\n".join(formatted_results).rstrip()


class BasicAgent:
    def __init__(self, repo: Path, client: InferenceClient):
        self.repo = repo
        self.client = client

        self.bugs = []
        self.updated_files = {}

        self.container_id = None
        self.container_name = f"basic_agent_{uuid.uuid4().hex[:8]}"
        self._file_history = {}

        # Register cleanup on exit
        atexit.register(self._cleanup_container)

        self.tools = [
            (
                self.bash,
                {
                    # https://www.anthropic.com/engineering/swe-bench-sonnet#:~:text=Next%2C%20we%20show%20the%20spec%20for%20the%20Bash%20Tool%3A
                    "name": "bash",
                    "description": textwrap.dedent(
                        """
                        Run commands in a bash shell
                        * When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
                        * You don't have access to the internet via this tool.
                        * You do have access to a mirror of common linux and python packages via apt and pip.
                        * State is persistent across command calls and discussions with the user.
                        * To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
                        * Please avoid commands that may produce a very large amount of output.
                        * Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
                        """
                    ).strip(),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The bash command to run.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            ),
            (
                self.str_replace_editor,
                {
                    # https://www.anthropic.com/engineering/swe-bench-sonnet#:~:text=The%20following%20code%20shows%20the%20description%20for%20our%20Edit%20Tool%3A
                    "name": "str_replace_editor",
                    "description": textwrap.dedent(
                        """
                        Custom editing tool for viewing, creating and editing files
                        * State is persistent across command calls and discussions with the user
                        * If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
                        * The `create` command cannot be used if the specified `path` already exists as a file
                        * If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
                        * The `undo_edit` command will revert the last edit made to the file at `path`

                        Notes for using the `str_replace` command:
                        * The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
                        * If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
                        * The `new_str` parameter should contain the edited lines that should replace the `old_str`
                        """
                    ).strip(),
                    "input_schema": {
                        # https://github.com/aws-samples/aws-mcp-servers-samples/blob/main/remote_computer_use/tools/tools_config.py#L101
                        "type": "object",
                        "properties": {
                            "command": {
                                "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                                "enum": [
                                    "view",
                                    "create",
                                    "str_replace",
                                    "insert",
                                    "undo_edit",
                                ],
                                "type": "string",
                            },
                            "file_text": {
                                "description": "Required parameter of `create` command, with the content of the file to be created.",
                                "type": "string",
                            },
                            "insert_line": {
                                "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                                "type": "integer",
                            },
                            "new_str": {
                                "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                                "type": "string",
                            },
                            "old_str": {
                                "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                                "type": "string",
                            },
                            "path": {
                                "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
                                "type": "string",
                            },
                            "view_range": {
                                "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                                "items": {"type": "integer"},
                                "type": "array",
                            },
                        },
                        "required": ["command", "path"],
                    },
                },
            ),
            (
                self.think,
                {
                    "name": "think",
                    "description": "Freely describe and reflect on what you know so far, things that you tried, and how that aligns with your objective and the user's intent. You can play through different scenarios, weigh options, and reason about possible next next steps. The user will not see any of your thoughts here, so you can think freely.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "thoughts": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["thoughts"],
                    },
                },
            ),
            (
                self.report_bug,
                {
                    "name": "report_bug",
                    "description": textwrap.dedent(
                        """
                        Report a bug in the codebase
                        * This tool is used to report bugs found during the review process.
                        * The `bug_description` parameter should contain a clear and concise description of the bug.
                        * The `file_path` parameter should point to the file where the bug was found.
                        * The `line_number` parameter should indicate the line number where the bug was found.
                        """
                    ).strip(),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "bug_description": {
                                "type": "string",
                                "description": "A clear and concise description of the bug.",
                            },
                            "file_path": {
                                "type": "string",
                                "description": "The path to the file where the bug was found.",
                            },
                            "line_number": {
                                "type": "integer",
                                "description": "The line number where the bug was found.",
                            },
                        },
                        "required": ["bug_description", "file_path", "line_number"],
                    },
                },
            ),
            (
                self.finish,
                {
                    "name": "finish",
                    "description": "Stop the review loop. Only use this when you're certain you have explored thoroughly and fixed all found bugs.",
                    "input_schema": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            ),
        ]

    async def bash(self, resp: Bash) -> str:
        """Execute bash commands in a persistent Docker container."""
        if not resp.command:
            return "Error: No command provided"

        try:
            # Ensure container is running
            await self._ensure_container()

            # Execute command in container with timeout
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                self.container_name,
                "bash",
                "-c",
                resp.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
                return stdout.decode("utf-8", errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Command timed out after 120 seconds"

        except Exception as e:
            return f"Error executing command: {str(e)}"

    async def _ensure_container(self):
        """Ensure the Docker container is running, creating it if necessary."""
        if self.container_id is None:
            # Create and start the container
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "-v",
                f"{self.repo}:{self.repo}",
                "-w",
                f"{self.repo}",
                "ubuntu:24.04",
                "tail",
                "-f",
                "/dev/null",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(f"Failed to create container: {stderr.decode()}")

            self.container_id = stdout.decode().strip()
        else:
            # Check if container is still running
            proc = await asyncio.create_subprocess_exec(
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}}",
                self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0 or stdout.decode().strip() != "true":
                # Container stopped, restart it
                await asyncio.create_subprocess_exec(
                    "docker",
                    "start",
                    self.container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )

    def _cleanup_container(self):
        """Clean up the Docker container when the agent is destroyed."""
        if self.container_id:
            try:
                subprocess.run(
                    ["docker", "kill", self.container_name],
                    capture_output=True,
                    timeout=10,
                )
                subprocess.run(
                    ["docker", "rm", self.container_name],
                    capture_output=True,
                    timeout=10,
                )
            except Exception:
                # Ignore cleanup errors
                pass

    def __del__(self):
        """Destructor to ensure container cleanup."""
        self._cleanup_container()

    # https://github.com/aws-samples/aws-mcp-servers-samples/blob/8b586ff5c837aadc499f0a1394a03b9e33315f7e/remote_computer_use/tools/edit.py
    async def str_replace_editor(self, resp: StrReplaceEditor) -> str:
        """
        Custom editing tool for viewing, creating and editing files
        """
        command = resp.command
        path = resp.path

        if not command:
            return "Error: No command provided"
        if not path:
            return "Error: No path provided"

        # Convert relative paths to absolute paths within the repo
        if not path.startswith("/"):
            path = str(self.repo / path)

        try:
            if command == "view":
                return await self._view_file(path, resp.view_range)
            elif command == "create":
                file_text = resp.file_text
                if file_text is None:
                    return (
                        "Error: Parameter `file_text` is required for command: create"
                    )
                return await self._create_file(path, file_text)
            elif command == "str_replace":
                old_str = resp.old_str
                new_str = resp.new_str or ""
                if old_str is None:
                    return "Error: Parameter `old_str` is required for command: str_replace"
                return await self._str_replace(path, old_str, new_str)
            elif command == "insert":
                insert_line = resp.insert_line
                new_str = resp.new_str
                if insert_line is None:
                    return (
                        "Error: Parameter `insert_line` is required for command: insert"
                    )
                if new_str is None:
                    return "Error: Parameter `new_str` is required for command: insert"
                return await self._insert_lines(path, insert_line, new_str)
            elif command == "undo_edit":
                return await self._undo_edit(path)
            else:
                return f"Error: Unrecognized command {command}. Allowed commands are: view, create, str_replace, insert, undo_edit"

        except Exception as e:
            return f"Error executing {command}: {str(e)}"

    async def _view_file(self, path: str, view_range: list = None) -> str:
        """View file or directory contents"""
        path_obj = Path(path)

        if not path_obj.exists():
            return f"Error: The path {path} does not exist"

        if path_obj.is_dir():
            if view_range:
                return "Error: The `view_range` parameter is not allowed when `path` points to a directory"

            # List directory contents up to 2 levels deep
            try:
                items = []
                for item in path_obj.rglob("*"):
                    # Calculate depth relative to the base path
                    depth = len(item.relative_to(path_obj).parts)
                    if depth <= 2 and not item.name.startswith("."):
                        items.append(str(item))

                items.sort()
                content = f"Here's the files and directories up to 2 levels deep in {path}, excluding hidden items:\n"
                content += "\n".join(items)
                return content
            except Exception as e:
                return f"Error listing directory: {str(e)}"

        # Handle file viewing
        try:
            with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()

            if view_range:
                if len(view_range) != 2 or not all(
                    isinstance(i, int) for i in view_range
                ):
                    return "Error: Invalid `view_range`. It should be a list of two integers"

                file_lines = file_content.split("\n")
                n_lines = len(file_lines)
                start_line, end_line = view_range

                if start_line < 1 or start_line > n_lines:
                    return f"Error: Invalid `view_range`: {view_range}. First element `{start_line}` should be within [1, {n_lines}]"

                if end_line != -1 and end_line > n_lines:
                    return f"Error: Invalid `view_range`: {view_range}. Second element `{end_line}` should be <= {n_lines}"

                if end_line != -1 and end_line < start_line:
                    return f"Error: Invalid `view_range`: {view_range}. Second element should be >= first element"

                if end_line == -1:
                    selected_lines = file_lines[start_line - 1 :]
                else:
                    selected_lines = file_lines[start_line - 1 : end_line]

                file_content = "\n".join(selected_lines)
                init_line = start_line
            else:
                init_line = 1

            return self._make_output(file_content, path, init_line)

        except Exception as e:
            return f"Error reading file: {str(e)}"

    async def _create_file(self, path: str, file_text: str) -> str:
        """Create a new file with the given content"""
        path_obj = Path(path)

        if path_obj.exists():
            return f"Error: File already exists at: {path}. Cannot overwrite files using command `create`"

        try:
            # Create parent directories if they don't exist
            path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(file_text)

            # Track this file for undo functionality
            self._file_history[path] = []

            # Track updated files
            self.updated_files[path] = file_text

            return f"File created successfully at: {path}"

        except Exception as e:
            return f"Error creating file: {str(e)}"

    async def _str_replace(self, path: str, old_str: str, new_str: str) -> str:
        """Replace old_str with new_str in the file"""
        path_obj = Path(path)

        if not path_obj.exists():
            return f"Error: The path {path} does not exist"

        if path_obj.is_dir():
            return f"Error: The path {path} is a directory and only the `view` command can be used on directories"

        try:
            # Read current content
            with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()

            # Expand tabs for consistent handling
            file_content = file_content.expandtabs()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs()

            # Check if old_str exists and is unique
            occurrences = file_content.count(old_str)
            if occurrences == 0:
                return f"Error: No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}"
            elif occurrences > 1:
                file_lines = file_content.split("\n")
                lines = [
                    idx + 1 for idx, line in enumerate(file_lines) if old_str in line
                ]
                return f"Error: No replacement was performed. Multiple occurrences of old_str `{old_str}` in lines {lines}. Please ensure it is unique"

            # Save current content to history for undo
            if path not in self._file_history:
                self._file_history[path] = []
            self._file_history[path].append(file_content)

            # Perform replacement
            new_file_content = file_content.replace(old_str, new_str)

            # Write the updated content
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(new_file_content)

            # Track updated files
            self.updated_files[path] = new_file_content

            # Create a snippet showing the change
            replacement_line = file_content.split(old_str)[0].count("\n")
            snippet_lines = 4
            start_line = max(0, replacement_line - snippet_lines)
            end_line = replacement_line + snippet_lines + new_str.count("\n")
            snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(
                snippet, f"a snippet of {path}", start_line + 1
            )
            success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

            return success_msg

        except Exception as e:
            return f"Error performing string replacement: {str(e)}"

    async def _insert_lines(self, path: str, insert_line: int, new_str: str) -> str:
        """Insert new_str after the specified line number"""
        path_obj = Path(path)

        if not path_obj.exists():
            return f"Error: The path {path} does not exist"

        if path_obj.is_dir():
            return f"Error: The path {path} is a directory and only the `view` command can be used on directories"

        try:
            # Read current content
            with open(path_obj, "r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()

            file_content = file_content.expandtabs()
            new_str = new_str.expandtabs()

            file_lines = file_content.split("\n")
            n_lines = len(file_lines)

            if insert_line < 0 or insert_line > n_lines:
                return f"Error: Invalid `insert_line` parameter: {insert_line}. It should be within [0, {n_lines}]"

            # Save current content to history for undo
            if path not in self._file_history:
                self._file_history[path] = []
            self._file_history[path].append(file_content)

            # Insert the new content
            new_str_lines = new_str.split("\n")
            new_file_lines = (
                file_lines[:insert_line] + new_str_lines + file_lines[insert_line:]
            )

            new_file_content = "\n".join(new_file_lines)

            # Write the updated content
            with open(path_obj, "w", encoding="utf-8") as f:
                f.write(new_file_content)

            # Track updated files
            self.updated_files[path] = new_file_content

            # Create a snippet showing the insertion
            snippet_lines = 4
            snippet_start = max(0, insert_line - snippet_lines)
            snippet_end = insert_line + len(new_str_lines) + snippet_lines
            snippet = "\n".join(new_file_lines[snippet_start:snippet_end])

            success_msg = f"The file {path} has been edited. "
            success_msg += self._make_output(
                snippet, "a snippet of the edited file", snippet_start + 1
            )
            success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

            return success_msg

        except Exception as e:
            return f"Error inserting lines: {str(e)}"

    async def _undo_edit(self, path: str) -> str:
        """Undo the last edit to the file"""
        if path not in self._file_history or not self._file_history[path]:
            return f"Error: No edit history found for {path}"

        try:
            # Get the previous version
            old_content = self._file_history[path].pop()

            # Write it back
            with open(path, "w", encoding="utf-8") as f:
                f.write(old_content)

            # Update tracked files
            self.updated_files[path] = old_content

            return f"Last edit to {path} undone successfully. {self._make_output(old_content, str(path))}"

        except Exception as e:
            return f"Error undoing edit: {str(e)}"

    def _make_output(
        self, file_content: str, file_descriptor: str, init_line: int = 1
    ) -> str:
        """Generate output for the CLI based on the content of a file"""
        # Truncate if too long
        max_length = 16000
        truncated_message = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with `grep -n` in order to find the line numbers of what you are looking for.</NOTE>"

        if len(file_content) > max_length:
            file_content = file_content[:max_length] + truncated_message

        file_content = file_content.expandtabs()

        # Add line numbers
        numbered_lines = []
        for i, line in enumerate(file_content.split("\n")):
            line_num = i + init_line
            numbered_lines.append(f"{line_num:6}\t{line}")

        numbered_content = "\n".join(numbered_lines)

        return f"Here's the result of running `cat -n` on {file_descriptor}:\n{numbered_content}\n"

    async def think(self, resp: Think) -> str:
        return str(resp.thoughts or ["No thoughts provided"])

    async def report_bug(self, resp: ReportBug) -> str:
        """Report a bug found in the codebase"""
        if not resp.bug_description:
            return "Error: No bug_description provided"
        if not resp.file_path:
            return "Error: No file_path provided"
        if resp.line_number is None:
            return "Error: No line_number provided"

        # Create bug report
        bug_report = {
            "description": resp.bug_description,
            "file": resp.file_path,
            "line": resp.line_number,
        }

        # Add to bugs list
        self.bugs.append(bug_report)

        return f"Bug reported: {resp.bug_description} at {resp.file_path}:{resp.line_number}"

    async def finish(self, _resp: Finish) -> str:
        raise StopAsyncIteration()

    async def process(self, subsystem):
        subsystem_name = subsystem["name"]
        files = ", ".join([f"'{file}'" for file in subsystem["files"]])

        content = interpolate_prompt(
            f"Please review the code in the {subsystem_name} subsystem (consisting of {files}) in the current directory for potential bugs.\n\nFocus on identifying issues that represent objectively incorrect behavior, could lead to exceptions or program crashes, or constitute security vulnerabilities.\n\n{{{{return_type}}}}\n\n",
            ToolTypes,
            format_tag="return_type",
        )
        for tool in (Bash, StrReplaceEditor, Think, ReportBug, Finish):
            content += f"\n\nDescription of {tool.__name__}:\n{tool.__doc__}\n\n"

        content += "Make one tool call then stop to see the result."

        await self.client.tool_chain(
            [
                ChatMessage(
                    role=ChatRole.USER,
                    content=content,
                )
            ],
            tools=self.tools,
            temperature=0.0,
            max_iterations=120,
            max_tokens=8192,
            tool_choice="any",
            fifo_ratio=0.7,
            tool_parser=tool_parser_func,
            tool_result_reducer=tool_result_reducer,
        )
        return self.bugs, self.updated_files


def read_urls(path: Path) -> Iterable[str]:
    logging.info(f"Reading URLs from {path}")
    with path.open() as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line.strip()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Bug-fix proximity evaluation using basic agent."
    )
    parser.add_argument("model", type=str, help="Model to use for inference.")
    parser.add_argument("url_file", type=Path, help="File with PR/commit URLs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("basic_agent_results"),
        help="Directory to store scan results (default: basic_agent_results)",
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
    args = parser.parse_args()

    # Set the diff cache directory
    DIFF_CACHE_DIR = args.diff_cache_dir
    DIFF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Create results directory
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving scan results to {results_dir}")

    urls = list(read_urls(args.url_file))
    if not urls:
        logging.error("No URLs to process.")
        sys.exit(1)

    # Shuffle URLs if requested
    if args.shuffle:
        logging.info(f"Shuffling {len(urls)} URLs")
        random.shuffle(urls)

    client = OpenRouterInferenceClient(
        model=args.model, api_key=os.environ["OPENROUTER_KEY"]
    )

    async def process_url(url: str, results_dir: Path, resume: bool = False):
        """
        Process a single PR or commit URL.
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
            checkout_sha = base_sha
        else:
            sha = commit_match["sha"]
            logging.info(f"Processing as commit {sha}")
            parent_sha, gold_map = get_commit_parent_and_line_map(owner, repo, sha)
            checkout_sha = parent_sha

        if not gold_map:
            logging.error(f"No changes found in {url}")
            return False, f"No changes found in {url}"

        scan_res = None
        if resume and result_exists(url, results_dir, prefix="basic_agent_scan"):
            logging.info(f"Found existing results for {url}, attempting to load")
            try:
                with open(
                    get_result_filename(url, results_dir, prefix="basic_agent_scan"),
                    "r",
                ) as f:
                    scan_res = json.load(f)
            except:
                logging.warning(
                    f"Failed to load existing results for {url}, re-scanning",
                    exc_info=True,
                )

        if scan_res is None:
            logging.info(f"Running basic agent scan on repository")

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

            # Create a temporary directory for the repository
            with tempfile.TemporaryDirectory() as tmp_dir:
                repo_dir = Path(tmp_dir)

                repo = Repo.clone_from(f"https://github.com/{repo_full}", tmp_dir)
                repo.git.checkout(checkout_sha)

                agent = BasicAgent(repo_dir, client)

                all_bugs = []
                all_updated_files = {}

                # Process each subsystem
                for i, subsystem in enumerate(subsystems):
                    logging.info(
                        f"Processing subsystem {i+1}/{len(subsystems)}: {subsystem['name']}"
                    )
                    bugs, updated_files = await agent.process(subsystem)
                    all_bugs.extend(bugs)
                    all_updated_files.update(updated_files)

                scan_res = {
                    "repo_id": repo_id,
                    "bugs": all_bugs,
                    "updated_files": all_updated_files,
                }

                # Save the scan results
                with open(
                    get_result_filename(url, results_dir, prefix="basic_agent_scan"),
                    "w",
                ) as f:
                    json.dump(scan_res, f)

        # Analyze scan results for overlap with gold standard
        logging.info(f"Analyzing scan results for overlap with gold standard")
        found = False
        details = []
        bug_count = len(scan_res.get("bugs", []))

        for bug in scan_res.get("bugs", []):
            file_path = bug.get("file", "")
            line_num = bug.get("line", 0)
            description = bug.get("description", "No description")

            # Check if the file is in the gold_map
            if file_path in gold_map:
                # Check if the line number is in the set of modified lines
                if line_num in gold_map[file_path]:
                    found = True
                    overlap_info = (
                        f"  Bug '{description}' overlaps '{file_path}' line {line_num}"
                    )
                    logging.info(f"Found overlap: {overlap_info}")
                    details.append(overlap_info)

        logging.info(f"Analysis complete: processed {bug_count} bugs")
        logging.info(f"Overlap detected: {found}")

        summary = "✅ Overlap detected." if found else "❌ No overlap."
        message = f"{url}\n{summary}"
        if details:
            message += "\n" + "\n".join(details)
        return found, message

    # Function to process a single URL for parallel execution
    def process_url_wrapper(url_index_tuple):
        index, url = url_index_tuple
        logging.info(f"Processing URL {index+1}/{len(urls)}: {url}")
        try:
            ok, msg = asyncio.run(process_url(url, results_dir, args.resume))
            return ok, msg, url, None  # None means no exception
        except KeyboardInterrupt:
            return False, "Interrupted", url, "KeyboardInterrupt"
        except Exception as e:
            logging.exception(f"Error processing {url}")
            return False, f"Error: {str(e)}", url, str(e)

    async def main():
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
                                    logging.error(
                                        f"Error processing {url}: {exception}"
                                    )
                                else:
                                    logging.info(f"No overlap found for {url}")
                        except Exception as e:
                            logging.exception(f"Error getting result for {u}")

                        # Update progress bar
                        progress_bar.update(1)
        else:
            # Sequential processing (original behavior)
            for i, url in enumerate(tqdm(urls, desc="Processing")):
                logging.info(f"Processing URL {i+1}/{len(urls)}: {url}")
                try:
                    ok, msg = await process_url(url, results_dir, args.resume)
                    print(msg)
                    if ok:
                        successes += 1
                        logging.info(f"Successfully found overlap for {url}")
                    else:
                        logging.info(f"No overlap found for {url}")
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    logging.exception(f"Error processing {url}")

        print("\n==== SUMMARY ====")
        print(f"Processed : {len(urls)}")
        print(f"Overlap   : {successes}")
        logging.info(
            f"Final results: processed {len(urls)} URLs, found overlap in {successes}"
        )

    asyncio.run(main())
