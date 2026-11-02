# Copyright 2026 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kernel-level operations layered on top of the proxied Colab frontend tools.

The notebook editing tools (`get_cells`, `run_code_cell`, `add_code_cell`, ...)
are served by the Colab frontend and reach us over the websocket bridge, so we
cannot change their signatures. Everything here is built *on* them from the
server side:

* `ProxyTools` calls a frontend tool by intent rather than by exact spelling —
  it reads the live input schema and maps our canonical argument names onto
  whatever the frontend actually declares.
* `ExecutionManager` turns the blocking `run_code_cell` into a launch/poll pair
  so a 15-minute cell no longer has to fit inside a client's request timeout.
* `KernelTools` composes those into the higher-level operations the agent
  actually wants: transient execution that leaves no cell behind, expression
  evaluation, batch runs, tailed output, and file transfer.

Nothing here mutates the notebook except where the tool name says so, and the
transient paths always clean up the cell they added.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

# Marker used to fish a JSON payload out of a cell's stdout. Transient helper
# code prints exactly one of these; anything else on stdout is the user's own
# output and is preserved separately.
PAYLOAD_BEGIN = "__COLAB_MCP_JSON_BEGIN__"
PAYLOAD_END = "__COLAB_MCP_JSON_END__"
_PAYLOAD_RE = re.compile(
    re.escape(PAYLOAD_BEGIN) + r"(.*?)" + re.escape(PAYLOAD_END), re.DOTALL
)

# Bytes of file content moved per transient cell. The base64 of this lands in a
# cell's stdout and travels over the websocket bridge; it never enters the
# agent's context, so the limit is about frontend/websocket comfort, not tokens.
FILE_CHUNK_BYTES = 256 * 1024
# Ceiling on a single transfer. The transport is chunked, so this is not a
# transport limit: it exists to refuse a transfer nobody meant to start before
# it spends hours of transient cells. The old 256 MB default was well under a
# single activation dump, which forced splitting files on the VM by hand;
# override with COLAB_MCP_MAX_TRANSFER_BYTES when the default is still wrong.
MAX_TRANSFER_BYTES = int(
    os.environ.get("COLAB_MCP_MAX_TRANSFER_BYTES", "") or 8 * 1024 * 1024 * 1024
)
# Log transfer progress every this many chunks (16 MB at the default chunk
# size), so a multi-gigabyte copy is visibly alive in the server log.
_TRANSFER_LOG_EVERY = 64

DEFAULT_TRANSIENT_TIMEOUT = 120.0
# A cell run we launched but never saw finish stays pollable this long before
# the manager stops treating it as live.
EXECUTION_RETENTION = 60 * 60.0

TERMINAL_STATES = ("done", "error", "interrupted", "timeout")

_MENU_FAILURE_NOTE = (
    "The Colab Runtime menu could not be driven. If the error names a dialog "
    "intercepting pointer events, call dismiss_colab_dialog and retry; "
    "screenshot_colab shows what is on screen."
)


class NotConnectedError(RuntimeError):
    """Raised when a kernel operation needs the Colab frontend and it is absent."""


def _too_large(path: str, size: int) -> str:
    return (
        f"{path} is {size} bytes; limit is {MAX_TRANSFER_BYTES} "
        f"({MAX_TRANSFER_BYTES // (1024 * 1024)} MB). The transfer is chunked, so the "
        "limit only guards against unintended multi-hour copies: raise it with "
        "COLAB_MCP_MAX_TRANSFER_BYTES in the server's environment."
    )


def _log_progress(direction: str, path: str, done: int, total: int) -> None:
    chunk_index = (done + FILE_CHUNK_BYTES - 1) // FILE_CHUNK_BYTES
    if done >= total or chunk_index % _TRANSFER_LOG_EVERY == 0:
        logging.info(
            "colab-mcp: %s %s: %d / %d bytes (%.0f%%)",
            direction, path, done, total, 100.0 * done / total if total else 100.0,
        )


def describe_error(exc: BaseException, limit: int = 400) -> str:
    """A one-paragraph account of `exc` fit for a tool result.

    Playwright's timeouts run to dozens of lines of call log; the useful part
    is the first line plus whichever line names the element that intercepted
    the click, so those are what survive.
    """
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    keep = lines[:1] + [
        line for line in lines[1:] if "intercepts pointer events" in line
    ]
    text = f"{type(exc).__name__}: " + " | ".join(keep) if keep else type(exc).__name__
    return text[:limit]


# --------------------------------------------------------------------------
# argument adaptation
# --------------------------------------------------------------------------

# Canonical argument name -> spellings a frontend might plausibly declare. We
# match against the live schema instead of hardcoding one, so a rename on the
# Colab side degrades to a logged warning rather than a broken tool.
ARG_ALIASES: dict[str, tuple[str, ...]] = {
    "cellId": ("cellId", "cell_id", "cellID", "id"),
    "content": ("content", "source", "code", "text", "cellContent", "cell_content"),
    "index": ("cellIndex", "index", "position", "insertAt", "insert_at", "atIndex"),
    "language": ("language", "lang", "cellLanguage"),
    "indexStart": ("cellIndexStart", "indexStart", "index_start", "start"),
    "indexEnd": ("cellIndexEnd", "indexEnd", "index_end", "end"),
    "includeOutputs": (
        "includeOutputs",
        "include_outputs",
        "withOutputs",
        "with_outputs",
        "outputs",
    ),
}

# Frontend tool name -> spellings we will accept for it.
TOOL_ALIASES: dict[str, tuple[str, ...]] = {
    "get_cells": ("get_cells", "getCells", "list_cells"),
    "run_code_cell": ("run_code_cell", "runCodeCell", "execute_cell", "run_cell"),
    "add_code_cell": ("add_code_cell", "addCodeCell", "insert_code_cell"),
    "update_cell": ("update_cell", "updateCell"),
    "delete_cell": ("delete_cell", "deleteCell", "remove_cell"),
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def adapt_args(schema: dict | None, args: dict[str, Any]) -> dict[str, Any]:
    """Rewrite canonical argument names to the ones `schema` declares.

    `None` values are dropped so callers can pass optional arguments
    unconditionally. When the schema exposes no properties (or is missing
    entirely) the arguments pass through untouched — better to let the frontend
    reject a call loudly than to silently drop what it may have required.
    """
    props = ((schema or {}).get("properties") or {}) if isinstance(schema, dict) else {}
    supplied = {k: v for k, v in args.items() if v is not None}
    if not props:
        return supplied

    by_norm = {_norm(p): p for p in props}
    out: dict[str, Any] = {}
    for key, value in supplied.items():
        target = None
        for candidate in ARG_ALIASES.get(key, (key,)):
            if candidate in props:
                target = candidate
                break
            if _norm(candidate) in by_norm:
                target = by_norm[_norm(candidate)]
                break
        if target is None:
            logging.warning(
                "colab-mcp: dropping argument %r; frontend tool declares %s",
                key,
                sorted(props),
            )
            continue
        out[target] = value
    return out


# --------------------------------------------------------------------------
# result / cell normalization
# --------------------------------------------------------------------------


def result_text(result: Any) -> str:
    """Concatenate the text blocks of a CallToolResult."""
    parts = []
    for block in getattr(result, "content", None) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def result_payload(result: Any) -> Any:
    """Best structured view of a tool result.

    FastMCP wraps a bare return value as `{"result": ...}`; unwrap that so
    callers see the frontend's own shape. Falls back to the text blocks, parsed
    as JSON when they happen to be JSON.
    """
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        if set(structured) == {"result"}:
            return structured["result"]
        return structured
    data = getattr(result, "data", None)
    if data is not None:
        return data
    text = result_text(result)
    with contextlib.suppress(ValueError):
        return json.loads(text)
    return text


def _first(mapping: dict, keys: tuple[str, ...], default=None):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    for key in keys:
        for actual in mapping:
            if _norm(actual) == _norm(key) and mapping[actual] is not None:
                return mapping[actual]
    return default


def normalize_cells(payload: Any) -> list[dict]:
    """Coerce whatever `get_cells` returned into a uniform cell list.

    Each entry is `{index, id, source, outputs, raw}`. Unknown shapes yield an
    empty list rather than raising — every caller has something sensible to say
    about "no cells".
    """
    cells = payload
    if isinstance(payload, dict):
        cells = _first(payload, ("cells", "result", "items", "data"), [])
    if isinstance(cells, str):
        with contextlib.suppress(ValueError):
            cells = json.loads(cells)
    if not isinstance(cells, list):
        return []

    out = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            out.append({"index": index, "id": None, "source": str(cell), "outputs": [], "raw": cell})
            continue
        source = _first(cell, ("source", "content", "code", "text"), "")
        if isinstance(source, list):
            source = "".join(source)
        outputs = _first(cell, ("outputs", "output", "results"), []) or []
        if not isinstance(outputs, list):
            outputs = [outputs]
        out.append(
            {
                "index": index,
                "id": _first(cell, ("cellId", "cell_id", "id", "cellID")),
                "source": source if isinstance(source, str) else str(source),
                "outputs": outputs,
                "raw": cell,
            }
        )
    return out


def _output_text(value: Any) -> str:
    if isinstance(value, list):
        return "".join(str(v) for v in value)
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)


def split_outputs(outputs: list) -> dict:
    """Separate a cell's outputs by kind.

    Only `display_data` is the kind colab-mcp reports stalely — progress bars in
    particular. Plain stdout/stderr was always current, and `execute_result` (a
    cell's return value) is written once when the cell finishes, so both are
    kept apart from the stale-prone bucket. Lumping them together would flag
    trustworthy output as untrustworthy.
    """
    stream: list[str] = []
    stderr: list[str] = []
    results: list[str] = []
    display: list[str] = []
    errors: list[str] = []

    def _plain(output) -> str:
        data = _first(output, ("data", "value"), {})
        if isinstance(data, dict):
            # text/plain is the one representation worth showing inline.
            return _output_text(data.get("text/plain")) or f"<{', '.join(sorted(data))}>"
        return _output_text(data)

    for output in outputs or []:
        if isinstance(output, str):
            stream.append(output)
            continue
        if not isinstance(output, dict):
            stream.append(str(output))
            continue

        kind = str(_first(output, ("output_type", "outputType", "type"), "") or "")
        if kind == "stream" or ("text" in output and "data" not in output):
            text = _output_text(_first(output, ("text", "data", "value"), ""))
            if str(output.get("name", "")).lower() == "stderr":
                stderr.append(text)
            else:
                stream.append(text)
        elif kind == "error" or "traceback" in output or "ename" in output:
            traceback = _output_text(output.get("traceback"))
            name = output.get("ename") or ""
            value = output.get("evalue") or ""
            errors.append(traceback or f"{name}: {value}".strip(": "))
        elif kind == "execute_result":
            results.append(_plain(output))
        else:
            display.append(_plain(output))

    return {
        "stream": "".join(stream),
        "stderr": "".join(stderr),
        "result": "\n".join(r for r in results if r),
        "display_data": "\n".join(d for d in display if d),
        "error": "\n".join(e for e in errors if e),
        "has_display_data": bool(display),
    }


EMPTY_OUTPUT = {
    "stream": "",
    "stderr": "",
    "result": "",
    "display_data": "",
    "error": "",
    "has_display_data": False,
}


def tail(text: str, max_lines: int | None = None, max_chars: int | None = None) -> str:
    """Last `max_lines` lines of `text`, then clipped to `max_chars`.

    Polling a long-running cell should cost a bounded number of tokens no matter
    how much the cell has printed since it started.
    """
    if max_lines is not None and max_lines >= 0:
        lines = text.splitlines()
        if len(lines) > max_lines:
            text = "\n".join(lines[-max_lines:])
    if max_chars is not None and max_chars >= 0 and len(text) > max_chars:
        text = text[-max_chars:]
    return text


def extract_payload(text: str) -> Any | None:
    """Pull the JSON payload a transient helper printed, if present."""
    match = _PAYLOAD_RE.search(text or "")
    if not match:
        return None
    with contextlib.suppress(ValueError):
        return json.loads(match.group(1))
    return None


def strip_payload(text: str) -> str:
    """`text` with the helper payload removed, leaving the user's own output."""
    return _PAYLOAD_RE.sub("", text or "").strip()


def strip_notebook_outputs(notebook: dict) -> dict:
    """Return `notebook` with every cell's outputs and execution counts cleared."""
    for cell in notebook.get("cells", []) or []:
        if isinstance(cell, dict) and cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    return notebook


# --------------------------------------------------------------------------
# proxied frontend tools
# --------------------------------------------------------------------------


class ProxyTools:
    """Thin, name-tolerant caller for the tools the Colab frontend exposes."""

    def __init__(self, proxy_client):
        # Accepts the client itself or a zero-arg callable returning it: the
        # session proxy only builds its client in start_proxy_server, well after
        # the tools are constructed.
        self._proxy_source = proxy_client
        self._schemas: dict[str, dict] | None = None
        self._schemas_for = None
        # cellId -> index, so a single cell's outputs can be fetched by range
        # instead of pulling the whole notebook down on every poll.
        self._index_cache: dict[str, int] = {}

    @property
    def proxy_client(self):
        if callable(self._proxy_source):
            return self._proxy_source()
        return self._proxy_source

    def client(self):
        proxy = self.proxy_client
        if proxy is None or not proxy.is_connected():
            raise NotConnectedError(
                "the Colab frontend is not connected; call "
                "open_colab_notebook or open_colab_browser_connection first"
            )
        return proxy.proxy_mcp_client

    def is_connected(self) -> bool:
        proxy = self.proxy_client
        return proxy is not None and proxy.is_connected()

    async def schemas(self) -> dict[str, dict]:
        """Input schemas of the frontend's tools, cached per connection.

        The cache is keyed on the client object so a reconnecting tab (which
        gets a fresh client) re-reads the schemas instead of trusting the old
        tab's.
        """
        client = self.client()
        if self._schemas is None or self._schemas_for is not client:
            try:
                tools = await client.list_tools()
            except Exception:  # noqa: BLE001
                logging.exception("colab-mcp: could not list frontend tools")
                return {}
            self._schemas = {t.name: (getattr(t, "inputSchema", None) or {}) for t in tools}
            self._schemas_for = client
        return self._schemas

    async def resolve_tool(self, canonical: str) -> str:
        """The frontend's actual name for `canonical`."""
        schemas = await self.schemas()
        if not schemas:
            return canonical
        by_norm = {_norm(name): name for name in schemas}
        for candidate in TOOL_ALIASES.get(canonical, (canonical,)):
            if candidate in schemas:
                return candidate
            if _norm(candidate) in by_norm:
                return by_norm[_norm(candidate)]
        return canonical

    async def call(self, canonical: str, timeout: float | None = None, **args) -> Any:
        """Call a frontend tool by canonical name; returns its CallToolResult."""
        client = self.client()
        name = await self.resolve_tool(canonical)
        schemas = await self.schemas()
        payload = adapt_args(schemas.get(name), args)
        logging.info("colab-mcp: proxy call %s(%s)", name, sorted(payload))
        return await client.call_tool(name, payload, timeout=timeout)

    async def get_cells(
        self,
        include_outputs: bool = False,
        index_start: int | None = None,
        index_end: int | None = None,
    ) -> list[dict]:
        """Fetch cells, optionally restricted to an index range.

        The range is what makes polling affordable: fetching one cell's outputs
        does not have to drag every other cell's source along with it.
        """
        result = await self.call(
            "get_cells",
            includeOutputs=include_outputs,
            indexStart=index_start,
            indexEnd=index_end,
        )
        cells = normalize_cells(result_payload(result))
        if index_start is None:
            # Only a full listing tells us the true index of every cell.
            self._index_cache = {
                str(c["id"]): c["index"] for c in cells if c["id"] is not None
            }
        else:
            for offset, cell in enumerate(cells):
                cell["index"] = index_start + offset
                if cell["id"] is not None:
                    self._index_cache[str(cell["id"])] = cell["index"]
        return cells

    async def cell_ids(self) -> list[Any]:
        return [c["id"] for c in await self.get_cells() if c["id"] is not None]

    async def index_of(self, cell_id: Any) -> int | None:
        """Index of `cell_id`, refreshing the cache once on a miss."""
        key = str(cell_id)
        if key not in self._index_cache:
            await self.get_cells()
        return self._index_cache.get(key)

    async def add_code_cell(
        self, code: str, index: int | None = None, language: str = "python"
    ) -> Any:
        """Insert a code cell (appending by default) and return its id.

        `cellIndex` and `language` are required by the frontend, so an append
        has to count the cells first. The reply's shape is the frontend's
        business: the id is read out of it defensively and, failing that,
        recovered by diffing the cell list around the call.
        """
        before_cells = await self.get_cells()
        before = {str(c["id"]) for c in before_cells if c["id"] is not None}
        if index is None:
            index = len(before_cells)
        result = await self.call(
            "add_code_cell", content=code, index=index, language=language
        )
        cell_id = _cell_id_from_result(result_payload(result))
        if cell_id is not None and str(cell_id) not in before:
            return cell_id
        for cell in await self.get_cells():
            if cell["id"] is not None and str(cell["id"]) not in before:
                return cell["id"]
        raise RuntimeError("could not determine the id of the newly added cell")

    async def delete_cell(self, cell_id: Any) -> None:
        await self.call("delete_cell", cellId=cell_id)
        self._index_cache.pop(str(cell_id), None)

    async def run_cell(self, cell_id: Any, timeout: float | None = None) -> Any:
        return await self.call("run_code_cell", cellId=cell_id, timeout=timeout)

    async def cell_with_outputs(self, cell_id: Any) -> dict | None:
        """One cell plus its outputs, fetched by index range where possible."""
        index = await self.index_of(cell_id)
        if index is not None:
            cells = await self.get_cells(
                include_outputs=True, index_start=index, index_end=index
            )
            for cell in cells:
                if cell["id"] is None or str(cell["id"]) == str(cell_id):
                    return cell
            # The cell moved since the cache was filled; fall through to a scan.
        for cell in await self.get_cells(include_outputs=True):
            if cell["id"] is not None and str(cell["id"]) == str(cell_id):
                return cell
        return None

    async def cell_output(self, cell_id: Any) -> dict:
        cell = await self.cell_with_outputs(cell_id)
        if cell is None:
            return dict(EMPTY_OUTPUT)
        return split_outputs(cell["outputs"])


def _cell_id_from_result(payload: Any) -> Any | None:
    if isinstance(payload, dict):
        return _first(
            payload, ("newCellId", "new_cell_id", "cellId", "cell_id", "id", "cellID")
        )
    if isinstance(payload, (str, int)) and str(payload).strip():
        text = str(payload).strip()
        # e.g. "added cell abc-123"
        match = re.search(r"[A-Za-z0-9_-]{4,}$", text)
        return match.group(0) if match else None
    return None


# --------------------------------------------------------------------------
# asynchronous cell execution
# --------------------------------------------------------------------------


@dataclass
class Execution:
    """One launched cell run, pollable after the launching call returned."""

    exec_id: str
    cell_id: Any
    state: str = "queued"
    started_at: float = field(default_factory=time.monotonic)
    running_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    task: asyncio.Task | None = None

    def elapsed_s(self) -> float:
        end = self.finished_at if self.finished_at is not None else time.monotonic()
        return round(end - self.started_at, 3)

    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def summary(self) -> dict:
        return {
            "execId": self.exec_id,
            "cellId": self.cell_id,
            "state": self.state,
            "elapsed_s": self.elapsed_s(),
            "error": self.error,
        }


class ExecutionManager:
    """Launches cell runs in the background and reports on them.

    `run_code_cell` blocks for as long as the cell takes, which for a model load
    or a sweep is far longer than a client will wait — and a client-side timeout
    reads as a *failure* even though the cell is running normally. Launching the
    call in a task and handing back a poll handle makes the timeout disappear:
    a long cell is simply `running`.

    Runs are serialized here as well as in the kernel, so `queued` versus
    `running` is a real distinction and the queue can be dropped on interrupt.
    """

    def __init__(self, tools: ProxyTools):
        self._tools = tools
        self._executions: dict[str, Execution] = {}
        self._counter = 0
        self._lock = asyncio.Lock()

    def _new_id(self) -> str:
        self._counter += 1
        return f"exec-{self._counter}"

    def get(self, handle: Any) -> Execution | None:
        """Look up by execId, or by cellId (most recent run of that cell)."""
        key = str(handle)
        if key in self._executions:
            return self._executions[key]
        matches = [e for e in self._executions.values() if str(e.cell_id) == key]
        return matches[-1] if matches else None

    def live(self) -> list[Execution]:
        return [e for e in self._executions.values() if not e.is_terminal()]

    def current(self) -> Execution | None:
        return next((e for e in self._executions.values() if e.state == "running"), None)

    def _prune(self) -> None:
        now = time.monotonic()
        stale = [
            key
            for key, ex in self._executions.items()
            if ex.is_terminal()
            and ex.finished_at is not None
            and now - ex.finished_at > EXECUTION_RETENTION
        ]
        for key in stale:
            del self._executions[key]

    async def launch(self, cell_id: Any) -> Execution:
        self._prune()
        execution = Execution(exec_id=self._new_id(), cell_id=cell_id)
        self._executions[execution.exec_id] = execution
        execution.task = asyncio.create_task(self._run(execution))
        # Let the task reach the lock so a run with a free kernel reports
        # "running" rather than a momentary, misleading "queued".
        await asyncio.sleep(0)
        return execution

    async def _run(self, execution: Execution) -> None:
        try:
            async with self._lock:
                if execution.state == "interrupted":
                    return  # dropped from the queue while waiting
                execution.state = "running"
                execution.running_at = time.monotonic()
                await self._tools.run_cell(execution.cell_id)
                # An interrupted cell still *returns* from run_code_cell (with a
                # KeyboardInterrupt in its outputs), so completing normally must
                # not overwrite the interrupted state recorded by interrupt().
                if execution.state == "running":
                    execution.state = "done"
        except asyncio.CancelledError:
            execution.state = "interrupted"
            raise
        except Exception as exc:  # noqa: BLE001
            execution.state = "error"
            execution.error = f"{type(exc).__name__}: {exc}"
            logging.exception("colab-mcp: execution %s failed", execution.exec_id)
        finally:
            if execution.finished_at is None and execution.state != "running":
                execution.finished_at = time.monotonic()

    async def status(self, handle: Any, tail_lines: int = 40, max_chars: int = 4000) -> dict:
        execution = self.get(handle)
        if execution is None:
            return {"state": "unknown", "error": f"no execution matching {handle!r}"}

        info = execution.summary()
        with contextlib.suppress(NotConnectedError, Exception):
            output = await self._tools.cell_output(execution.cell_id)
            info["output_tail"] = tail(output["stream"], tail_lines, max_chars)
            info["stderr"] = tail(output["stderr"], tail_lines, max_chars)
            info["display_data"] = tail(output["display_data"], tail_lines, max_chars)
            if output["error"]:
                info["cell_error"] = tail(output["error"], tail_lines, max_chars)
                # A KeyboardInterrupt traceback is the *expected* result of an
                # interrupt, not a failure, so only a clean run is downgraded.
                if execution.state == "done":
                    info["state"] = "error"
        return info

    def drop_pending(self) -> list[str]:
        """Cancel runs we have queued but not yet handed to the kernel.

        Interrupting the kernel stops the cell it is on; without this the next
        queued cell would start immediately, which is the opposite of what
        someone killing a sweep wants.
        """
        dropped = []
        for execution in list(self._executions.values()):
            if execution.state == "queued":
                execution.state = "interrupted"
                execution.finished_at = time.monotonic()
                if execution.task is not None:
                    execution.task.cancel()
                dropped.append(execution.exec_id)
        return dropped

    def mark_interrupted(self) -> list[str]:
        """Record that the kernel was interrupted out from under a running cell."""
        marked = []
        for execution in self._executions.values():
            if execution.state == "running":
                execution.state = "interrupted"
                execution.finished_at = time.monotonic()
                marked.append(execution.exec_id)
        return marked


# --------------------------------------------------------------------------
# high-level kernel operations
# --------------------------------------------------------------------------


class KernelTools:
    """The operations the injected MCP tools delegate to."""

    def __init__(self, proxy_client, browser=None):
        self.tools = ProxyTools(proxy_client)
        self.executions = ExecutionManager(self.tools)
        self.browser = browser
        # Set by a successful delete_runtime and cleared by the next execution:
        # while it holds, anything that would run code on the VM must know it
        # is about to provision a fresh runtime, not inspect the old one.
        self._runtime_deleted = False

    # -- execution ---------------------------------------------------------

    async def run_cell_async(self, cell_id: Any) -> dict:
        self._runtime_deleted = False
        execution = await self.executions.launch(cell_id)
        return execution.summary()

    async def execution_status(self, handle: Any, tail_lines: int = 40) -> dict:
        return await self.executions.status(handle, tail_lines=tail_lines)

    async def run_cells(self, cell_ids: list[Any], stop_on_error: bool = True) -> list[dict]:
        """Run cells in order, reporting each one's outcome.

        Rebuilding kernel state after a runtime switch is the motivating case:
        one call instead of six, and it stops at the first failure by default so
        the tail of the list does not run against a broken state.
        """
        self._runtime_deleted = False
        results = []
        for cell_id in cell_ids:
            entry: dict[str, Any] = {"cellId": cell_id}
            try:
                await self.tools.run_cell(cell_id)
                output = await self.tools.cell_output(cell_id)
                entry["state"] = "error" if output["error"] else "done"
                entry["error"] = output["error"] or None
                entry["output_tail"] = tail(output["stream"], 20, 2000)
            except Exception as exc:  # noqa: BLE001
                entry["state"] = "error"
                entry["error"] = f"{type(exc).__name__}: {exc}"
            results.append(entry)
            if stop_on_error and entry["state"] == "error":
                remaining = cell_ids[len(results):]
                results.extend({"cellId": c, "state": "skipped"} for c in remaining)
                break
        return results

    async def run_all_below(self, cell_id: Any, stop_on_error: bool = True) -> list[dict]:
        cells = await self.tools.get_cells()
        ids = [c["id"] for c in cells if c["id"] is not None]
        try:
            start = [str(i) for i in ids].index(str(cell_id))
        except ValueError:
            return [{"cellId": cell_id, "state": "error", "error": "cell not found"}]
        return await self.run_cells(ids[start:], stop_on_error=stop_on_error)

    # -- transient execution ----------------------------------------------

    async def execute_transient(
        self, code: str, timeout_s: float = DEFAULT_TRANSIENT_TIMEOUT
    ) -> dict:
        """Run `code` in the kernel without leaving a cell behind.

        Debugging one-offs — a memory probe, a sign-convention check, a
        `nvidia-smi` — do not belong in a notebook meant to be shared, but the
        only way to reach the kernel is through a cell. So: add, run, read,
        delete. The delete runs even when the code raises or times out.
        """
        self._runtime_deleted = False
        cell_id = await self.tools.add_code_cell(code)
        timed_out = False
        error: str | None = None
        try:
            try:
                await asyncio.wait_for(self.tools.run_cell(cell_id), timeout=timeout_s)
            except asyncio.TimeoutError:
                timed_out = True
            output = await self.tools.cell_output(cell_id)
        except Exception as exc:  # noqa: BLE001
            output = dict(EMPTY_OUTPUT)
            error = f"{type(exc).__name__}: {exc}"
        finally:
            # Best effort: a cell we cannot delete is a smaller problem than a
            # tool that raises on cleanup and loses the output it just captured.
            with contextlib.suppress(Exception):
                await self.tools.delete_cell(cell_id)

        stdout = output["stream"]
        return {
            "stdout": strip_payload(stdout) if extract_payload(stdout) else stdout,
            "stderr": output["stderr"],
            "result": output["result"],
            "error": error or output["error"] or None,
            "payload": extract_payload(stdout),
            "timed_out": timed_out,
        }

    async def _run_payload(self, code: str, timeout_s: float) -> dict:
        """execute_transient, insisting on a JSON payload coming back."""
        result = await self.execute_transient(code, timeout_s=timeout_s)
        payload = result.get("payload")
        if payload is None:
            detail = result.get("error") or result.get("stdout") or "no output"
            if result.get("timed_out"):
                detail = f"timed out after {timeout_s}s"
            raise RuntimeError(f"kernel helper returned no payload: {detail}")
        return payload

    async def eval_expression(
        self, expr: str, max_repr_chars: int = 4000, timeout_s: float = 30.0
    ) -> dict:
        """Evaluate `expr` and report its repr, type, and shape if it has one.

        Reads a variable without editing a cell to add a `print` and re-running
        it — the expression is evaluated, never assigned, and the repr is
        truncated in the kernel so a big tensor cannot flood the transport.
        """
        code = _EVAL_TEMPLATE.format(
            expr=json.dumps(expr),
            limit=int(max_repr_chars),
            begin=json.dumps(PAYLOAD_BEGIN),
            end=json.dumps(PAYLOAD_END),
        )
        return await self._run_payload(code, timeout_s)

    async def runtime_attached(self) -> bool | None:
        """Whether a runtime is attached, without running code to find out.

        The browser's toolbar is the authority when it can be read; otherwise
        the memory of our own delete_runtime stands until something executes.
        None means genuinely unknown.
        """
        probe = None
        if self.browser is not None:
            with contextlib.suppress(Exception):
                probe = await self.browser.runtime_attached()
        if probe is True or probe is False:
            # Trust the live UI over our memory in both directions: the user
            # may have reconnected by hand, or disconnected from the toolbar.
            self._runtime_deleted = not probe
            return probe
        if self._runtime_deleted:
            return False
        return None

    async def runtime_info(self, timeout_s: float = 60.0) -> dict:
        """Hardware and resource state of the attached runtime.

        This reads as a status check, so it must behave like one: probing the
        VM means running a cell, and running a cell on a notebook with no
        runtime attached provisions a new one and restarts billing. When we
        know nothing is attached, say so instead.
        """
        if await self.runtime_attached() is False:
            return {
                "connected": self.tools.is_connected(),
                "runtime_attached": False,
                "note": (
                    "no runtime is attached to this notebook, so nothing was "
                    "probed and no VM was started. Running code (execute_transient, "
                    "run_cells, upload_file, ...) will provision a fresh runtime."
                ),
            }
        info = await self._run_payload(
            _RUNTIME_INFO_TEMPLATE.format(
                begin=json.dumps(PAYLOAD_BEGIN), end=json.dumps(PAYLOAD_END)
            ),
            timeout_s,
        )
        info["connected"] = self.tools.is_connected()
        info["runtime_attached"] = True
        return info

    # -- inspection --------------------------------------------------------

    async def get_cell_output(
        self, cell_id: Any, tail_lines: int = 50, max_chars: int = 8000
    ) -> dict:
        """Tail one cell's output, flagging whether the rich output is trustworthy.

        `get_cells(includeOutputs=True)` returns a stale snapshot of
        `display_data` — a progress bar can read `0/398` for ten minutes while
        the cell is nearly done — but plain stdout/stderr was always current.
        `display_data_current` says which of those you are looking at, so nobody
        has to fall back to a screenshot to find out.
        """
        cell = await self.tools.cell_with_outputs(cell_id)
        if cell is not None:
            output = split_outputs(cell["outputs"])
            execution = self.executions.get(cell_id)
            running = execution is not None and execution.state == "running"
            return {
                "cellId": cell["id"] if cell["id"] is not None else cell_id,
                "stream": tail(output["stream"], tail_lines, max_chars),
                "stderr": tail(output["stderr"], tail_lines, max_chars),
                "result": tail(output["result"], tail_lines, max_chars),
                "display_data": tail(output["display_data"], tail_lines, max_chars),
                "error": output["error"] or None,
                # Rich output is only reliable once the cell has stopped
                # producing it.
                "display_data_current": bool(
                    output["has_display_data"] and not running
                )
                if output["has_display_data"]
                else True,
                "running": running,
            }
        return {"cellId": cell_id, "error": "cell not found"}

    async def notebook_outline(self, source_chars: int = 120) -> list[dict]:
        """Cell ids with a source snippet and output sizes — no full bodies.

        Polling a long cell with `get_cells` re-sends every cell's entire source
        *and* every accumulated output on each check. This is the cheap view for
        "which cell is which".
        """
        outline = []
        for cell in await self.tools.get_cells(include_outputs=True):
            output = split_outputs(cell["outputs"])
            snippet = cell["source"].strip().splitlines()
            outline.append(
                {
                    "index": cell["index"],
                    "cellId": cell["id"],
                    "source_head": (snippet[0][:source_chars] if snippet else ""),
                    "source_lines": len(snippet),
                    "output_chars": len(output["stream"]) + len(output["display_data"]),
                    "has_error": bool(output["error"]),
                }
            )
        return outline

    async def kernel_status(self) -> dict:
        """Whether a newly submitted cell will run now or sit behind a long job.

        Executions this server launched are known exactly; a cell the agent ran
        through the raw `run_code_cell` tool is not, so the frontend's own cell
        state is consulted as a second source and `tracked` says which answer
        you are getting.
        """
        if not self.tools.is_connected():
            return {"state": "disconnected", "queue_depth": 0, "tracked": True}

        current = self.executions.current()
        queue_depth = len(self.executions.live())
        if current is not None:
            return {
                "state": "busy",
                "queue_depth": queue_depth,
                "current_cellId": current.cell_id,
                "current_execId": current.exec_id,
                "elapsed_s": current.elapsed_s(),
                "tracked": True,
            }

        busy_cell = None
        with contextlib.suppress(Exception):
            for cell in await self.tools.get_cells():
                raw = cell["raw"] if isinstance(cell["raw"], dict) else {}
                state = str(_first(raw, ("state", "status", "executionState"), "") or "")
                if state.lower() in ("running", "executing", "busy") or _first(
                    raw, ("executing", "isRunning", "running")
                ):
                    busy_cell = cell["id"]
                    break
        if busy_cell is not None:
            return {
                "state": "busy",
                "queue_depth": queue_depth,
                "current_cellId": busy_cell,
                "tracked": False,
            }
        return {"state": "idle", "queue_depth": queue_depth, "tracked": True}

    # -- control -----------------------------------------------------------

    async def interrupt(self, cancel_pending: bool = True) -> dict:
        """Interrupt the running cell without restarting the kernel.

        This is a SIGINT to the kernel via Colab's own Runtime menu, not
        `set_gpu_runtime`'s restart: globals assigned before the interrupt — a
        multi-gigabyte model in particular — survive it.
        """
        current = self.executions.current()
        tracked_running = current is not None
        was_running = tracked_running
        if not was_running:
            status = await self.kernel_status()
            was_running = status.get("state") == "busy"

        dropped = self.executions.drop_pending() if cancel_pending else []
        if self.browser is None:
            return {
                "interrupted": False,
                "was_running": was_running,
                "cancelled_pending": dropped,
                "error": "no browser session; call open_colab_browser_connection first",
            }

        interrupted = await self.browser.interrupt_execution()
        if interrupted:
            self.executions.mark_interrupted()
        return {
            "interrupted": bool(interrupted),
            "was_running": was_running,
            "cancelled_pending": dropped,
        }

    async def restart_kernel(self) -> dict:
        if self.browser is None:
            return {"restarted": False, "error": "no browser session"}
        self.executions.drop_pending()
        self.executions.mark_interrupted()
        try:
            restarted = await self.browser.restart_kernel()
        except Exception as exc:  # noqa: BLE001
            logging.exception("colab-mcp: could not drive the Runtime menu to restart")
            return {
                "restarted": False,
                "error": describe_error(exc),
                "note": _MENU_FAILURE_NOTE,
            }
        return {"restarted": bool(restarted)}

    async def delete_runtime(self) -> dict:
        """Disconnect and delete the runtime VM, not just the kernel process.

        Everything tracked in-flight is dropped first: the VM is going away, so
        any pending execution can never report back, and leaving those entries
        would strand get_execution_status on ids that will never resolve.
        """
        if self.browser is None:
            return {"deleted": False, "error": "no browser session"}
        cancelled = self.executions.drop_pending()
        self.executions.mark_interrupted()
        try:
            deleted = await self.browser.delete_runtime()
        except Exception as exc:  # noqa: BLE001
            # Never fold this into "is a runtime connected?": the menu was
            # unreachable, the runtime is most likely still up and billing,
            # and the caller needs the real obstacle to clear it.
            logging.exception("colab-mcp: could not drive the Runtime menu to delete")
            return {
                "deleted": False,
                "cancelled_pending": cancelled,
                "error": describe_error(exc),
                "note": (
                    "the runtime was NOT deleted and may still be running and "
                    "billing. " + _MENU_FAILURE_NOTE
                ),
            }
        if deleted:
            self._runtime_deleted = True
        return {
            "deleted": bool(deleted),
            "cancelled_pending": cancelled,
            "note": (
                "runtime VM deleted; kernel state and files under /content are gone. "
                "The next execution starts a fresh runtime."
                if deleted
                else "the Runtime menu opened but had no delete-runtime entry; "
                "is a runtime connected?"
            ),
        }

    # -- file transfer -----------------------------------------------------

    async def export_notebook(
        self, dest_path: str, include_outputs: bool = True
    ) -> dict:
        """Save the executed notebook to local disk via the browser's download.

        The bytes go browser → disk directly, so a 250 KB notebook costs nothing
        in context, and there is no dependence on Drive sync timing. The
        notebook is saved first so what lands is not a stale autosave.
        """
        if self.browser is None:
            return {"error": "no browser session; call open_colab_browser_connection first"}
        saved = await self.browser.save_notebook()
        path = await self.browser.download_notebook(dest_path)

        cells = None
        if not include_outputs:
            with open(path, encoding="utf-8") as handle:
                notebook = json.load(handle)
            notebook = strip_notebook_outputs(notebook)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(notebook, handle, indent=1)
        with contextlib.suppress(Exception):
            with open(path, encoding="utf-8") as handle:
                cells = len(json.load(handle).get("cells", []))

        return {
            "path": path,
            "bytes": os.path.getsize(path),
            "cells": cells,
            # A failed save means the download may predate recent edits.
            "saved_before_export": bool(saved),
        }

    async def download_file(self, remote_path: str, dest_path: str) -> dict:
        """Copy a file off the Colab VM to local disk, in chunks via the kernel.

        The base64 travels over the websocket bridge and is written straight to
        disk here; it never passes through the agent's context.
        """
        meta = await self._run_payload(
            _STAT_TEMPLATE.format(
                path=json.dumps(remote_path),
                begin=json.dumps(PAYLOAD_BEGIN),
                end=json.dumps(PAYLOAD_END),
            ),
            60.0,
        )
        if meta.get("error"):
            return {"error": meta["error"]}
        size = int(meta["size"])
        if size > MAX_TRANSFER_BYTES:
            return {"error": _too_large(remote_path, size)}

        os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
        written = 0
        with open(dest_path, "wb") as handle:
            for offset in range(0, size, FILE_CHUNK_BYTES):
                chunk = await self._run_payload(
                    _READ_CHUNK_TEMPLATE.format(
                        path=json.dumps(remote_path),
                        offset=offset,
                        size=FILE_CHUNK_BYTES,
                        begin=json.dumps(PAYLOAD_BEGIN),
                        end=json.dumps(PAYLOAD_END),
                    ),
                    120.0,
                )
                data = base64.b64decode(chunk["b64"])
                handle.write(data)
                written += len(data)
                _log_progress("download", remote_path, written, size)
        return {"path": dest_path, "bytes": written, "remote_bytes": size}

    async def upload_file(self, src_path: str, remote_path: str) -> dict:
        """Copy a local file onto the Colab VM, in chunks via the kernel."""
        size = os.path.getsize(src_path)
        if size > MAX_TRANSFER_BYTES:
            return {"error": _too_large(src_path, size)}

        sent = 0
        with open(src_path, "rb") as handle:
            first = True
            while True:
                chunk = handle.read(FILE_CHUNK_BYTES)
                if not chunk and not first:
                    break
                await self._run_payload(
                    _WRITE_CHUNK_TEMPLATE.format(
                        path=json.dumps(remote_path),
                        b64=json.dumps(base64.b64encode(chunk).decode()),
                        mode=json.dumps("wb" if first else "ab"),
                        begin=json.dumps(PAYLOAD_BEGIN),
                        end=json.dumps(PAYLOAD_END),
                    ),
                    120.0,
                )
                sent += len(chunk)
                first = False
                _log_progress("upload", remote_path, sent, size)
                if not chunk:
                    break
        return {"remote_path": remote_path, "bytes": sent}


# --------------------------------------------------------------------------
# kernel-side helper snippets
# --------------------------------------------------------------------------

# Each prints exactly one delimited JSON payload so the result can be parsed out
# of stdout without disturbing whatever else the cell wrote.

_EVAL_TEMPLATE = """
import json as _j
_out = {{}}
try:
    _v = eval(compile({expr}, "<colab-mcp-eval>", "eval"))
    _r = repr(_v)
    _out["repr"] = _r[:{limit}]
    _out["truncated"] = len(_r) > {limit}
    _out["type"] = type(_v).__name__
    _s = getattr(_v, "shape", None)
    if _s is not None:
        _out["shape"] = list(_s) if hasattr(_s, "__iter__") else str(_s)
    _d = getattr(_v, "dtype", None)
    if _d is not None:
        _out["dtype"] = str(_d)
    try:
        _out["len"] = len(_v)
    except Exception:
        pass
except Exception as _e:
    _out["error"] = "%s: %s" % (type(_e).__name__, _e)
print({begin} + _j.dumps(_out) + {end})
"""

_RUNTIME_INFO_TEMPLATE = """
import json as _j, os as _os, shutil as _sh, subprocess as _sp
_out = {{"accelerator": "CPU"}}
try:
    _n = _sp.run(
        ["nvidia-smi",
         "--query-gpu=name,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20)
    if _n.returncode == 0 and _n.stdout.strip():
        _name, _tot, _used = [x.strip() for x in _n.stdout.strip().splitlines()[0].split(",")]
        _out.update(accelerator="GPU", gpu_name=_name,
                    vram_total_mb=int(float(_tot)), vram_used_mb=int(float(_used)))
except Exception as _e:
    _out["gpu_error"] = str(_e)
try:
    import torch as _t
    _out["torch"] = _t.__version__
    if _t.cuda.is_available():
        _out["accelerator"] = "GPU"
        _out["gpu_name"] = _t.cuda.get_device_name(0)
        _out["torch_reserved_mb"] = int(_t.cuda.memory_reserved(0) / 2**20)
        _out["torch_allocated_mb"] = int(_t.cuda.memory_allocated(0) / 2**20)
except Exception:
    pass
try:
    _mt = {{}}
    for _line in open("/proc/meminfo"):
        _k, _v = _line.split(":", 1)
        _mt[_k] = int(_v.strip().split()[0])
    _out["ram_total_mb"] = _mt["MemTotal"] // 1024
    _out["ram_available_mb"] = _mt.get("MemAvailable", 0) // 1024
except Exception:
    pass
try:
    _u = _sh.disk_usage("/content")
    _out["disk_total_gb"] = round(_u.total / 2**30, 2)
    _out["disk_free_gb"] = round(_u.free / 2**30, 2)
except Exception:
    pass
_out["hostname"] = _os.uname().nodename
print({begin} + _j.dumps(_out) + {end})
"""

_STAT_TEMPLATE = """
import json as _j, os as _os
_p = {path}
_out = {{}}
if _os.path.isfile(_p):
    _out["size"] = _os.path.getsize(_p)
else:
    _out["error"] = "no such file: %s" % _p
    _out["size"] = 0
print({begin} + _j.dumps(_out) + {end})
"""

_READ_CHUNK_TEMPLATE = """
import base64 as _b, json as _j
with open({path}, "rb") as _f:
    _f.seek({offset})
    _chunk = _f.read({size})
print({begin} + _j.dumps({{"b64": _b.b64encode(_chunk).decode()}}) + {end})
"""

_WRITE_CHUNK_TEMPLATE = """
import base64 as _b, json as _j, os as _os
_p = {path}
_d = _os.path.dirname(_p)
if _d:
    _os.makedirs(_d, exist_ok=True)
with open(_p, {mode}) as _f:
    _n = _f.write(_b.b64decode({b64}))
print({begin} + _j.dumps({{"written": _n}}) + {end})
"""
