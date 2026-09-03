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

import asyncio
from collections.abc import AsyncIterator
import contextlib
from contextlib import AsyncExitStack
from fastmcp import FastMCP, Client
from fastmcp.client.transports import ClientTransport
from fastmcp.dependencies import CurrentContext
from fastmcp.server.context import Context
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp.server.middleware.tool_injection import ToolInjectionMiddleware
from fastmcp.server.proxy import FastMCPProxy
from fastmcp.tools.tool import Tool, ToolResult
from mcp.client.session import ClientSession
from mcp.types import TextContent
import json
import logging
import os
import re
import tempfile
import threading
from urllib.parse import urlparse
import webbrowser

from colab_mcp.browser import ColabBrowser
from colab_mcp.kernel import KernelTools, NotConnectedError
from colab_mcp.websocket_server import (
    ColabWebSocketServer,
    COLAB,
    COLAB_ALT_DOMAIN,
    SCRATCH_PATH,
)

UI_CONNECTION_TIMEOUT = 60.0  # secs
# After navigating, how long to let the previous tab's websocket drop before we
# start treating a live connection as the new tab's.
STALE_DISCONNECT_TIMEOUT = 5.0  # secs

FE_CONNECTED_KEY = "fe_connected"
PROXY_TOKEN_KEY = "proxy_token"
PROXY_PORT_KEY = "proxy_port"
INJECTED_TOOL_NAME = "open_colab_browser_connection"
OPEN_NOTEBOOK_TOOL_NAME = "open_colab_notebook"

_COLAB_HOSTS = {urlparse(COLAB).netloc, urlparse(COLAB_ALT_DOMAIN).netloc}
_DRIVE_HOSTS = {"drive.google.com", "docs.google.com"}
# Drive file ids are long base64url-ish strings; short values are far more
# likely to be a typo than an id, so require some length before assuming one.
_DRIVE_ID_RE = re.compile(r"[A-Za-z0-9_-]{15,}")
_SCRATCH_ALIASES = {"", "new", "scratch", "scratchpad", "empty"}


def colab_notebook_path(notebook: str) -> str:
    """Resolve a user-supplied notebook reference to a Colab path.

    Accepts a Colab URL, a Drive file id or Drive/Docs URL, a GitHub blob or
    gist URL, a bare Colab path, or an empty/"new" alias for the scratch
    notebook. Any fragment on an incoming URL is dropped — see
    `colab_notebook_url` for why that matters.
    """
    ref = (notebook or "").strip()
    if ref.lower() in _SCRATCH_ALIASES:
        return SCRATCH_PATH
    if _DRIVE_ID_RE.fullmatch(ref):
        return f"/drive/{ref}"
    if ref.startswith("/"):
        return ref

    parsed = urlparse(ref if "//" in ref else f"https://{ref}")
    host = (parsed.netloc or "").lower().removeprefix("www.")
    path = parsed.path or ""
    query = f"?{parsed.query}" if parsed.query else ""

    if host in _COLAB_HOSTS:
        return f"{path}{query}" if path else SCRATCH_PATH
    if host in _DRIVE_HOSTS:
        # .../d/<id>/edit, or an ?id=<id> query on the older export URLs.
        match = re.search(r"/d/([A-Za-z0-9_-]+)", path) or re.search(
            r"[?&]id=([A-Za-z0-9_-]+)", ref
        )
        if match:
            return f"/drive/{match.group(1)}"
        raise ValueError(f"no Drive file id found in {notebook!r}")
    if host == "github.com":
        return f"/github{path}"
    if host == "gist.github.com":
        return f"/gist{path}"
    raise ValueError(
        f"unrecognised notebook reference {notebook!r}: expected a Colab, "
        "Drive, GitHub or gist URL, a Drive file id, or a Colab path"
    )


def colab_notebook_url(notebook: str, token: str, port: int) -> str:
    """Build the tokened Colab URL for `notebook`.

    The bridge handshake lives entirely in the URL fragment, so a notebook
    opened without it renders fine but never connects back to this server — and
    a URL copied from the address bar of an earlier session carries a *stale*
    token. Both cases are handled by discarding any inbound fragment and
    appending the current one.
    """
    return (
        f"{COLAB}{colab_notebook_path(notebook)}"
        f"#mcpProxyToken={token}&mcpProxyPort={port}"
    )


def _log_task_exception(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    if exc := task.exception():
        logging.error("colab-mcp: background task failed", exc_info=exc)


class ColabTransport(ClientTransport):
    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss

    @contextlib.asynccontextmanager
    async def connect_session(self, **session_kwargs) -> AsyncIterator[ClientSession]:
        async with ClientSession(
            self.wss.read_stream, self.wss.write_stream, **session_kwargs
        ) as session:
            yield session

    def __repr__(self) -> str:
        return "<ColabSessionProxyTransport>"


class ColabProxyClient:
    """Maintains an MCP client session against whichever Colab frontend is
    currently connected to the websocket bridge.

    A manager task waits for a frontend connection, performs the MCP handshake
    over that connection's streams, and tears the session down again when the
    frontend disconnects — so each reconnecting Colab tab gets a fresh
    initialize handshake instead of colliding with a stale session.
    """

    def __init__(self, wss: ColabWebSocketServer):
        self.wss = wss
        self.stubbed_mcp_client = Client(FastMCP())
        self.proxy_mcp_client: Client | None = None
        self.client_ready = asyncio.Event()
        self._manager_task: asyncio.Task | None = None

    def is_connected(self):
        return self.wss.connection_live.is_set() and self.proxy_mcp_client is not None

    async def await_proxy_connection(self):
        # The manager task owns the client lifecycle, so cancelling this wait
        # (e.g. the client's request timeout aborting the tool call) can't tear
        # the proxy client down.
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                self.client_ready.wait(), timeout=UI_CONNECTION_TIMEOUT
            )

    def client_factory(self):
        if self.is_connected():
            return self.proxy_mcp_client
        # return a client mapped to a stubbed mcp server if there is no session proxy
        return self.stubbed_mcp_client

    async def _run(self):
        while True:
            await self.wss.connection_live.wait()
            try:
                async with Client(ColabTransport(self.wss)) as proxy_client:
                    self.proxy_mcp_client = proxy_client
                    self.client_ready.set()
                    await self.wss.connection_closed.wait()
            except Exception:
                logging.exception("colab-mcp: proxy client session failed")
                # avoid a hot retry loop if the handshake keeps failing while
                # the websocket connection stays up
                await asyncio.sleep(1)
            finally:
                self.client_ready.clear()
                self.proxy_mcp_client = None

    async def __aenter__(self):
        self._manager_task = asyncio.create_task(self._run())
        self._manager_task.add_done_callback(_log_task_exception)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._manager_task:
            self._manager_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._manager_task


class ColabProxyMiddleware(Middleware):
    def __init__(self, proxy_client: ColabProxyClient):
        self.proxy_client = proxy_client
        self.last_message_connected = self.proxy_client.is_connected()

    async def on_message(self, context: MiddlewareContext, call_next):
        """
        Check for a change to Colab session connectivity on any communication with this MCP server and
        notify the client when the connectivity status has changed.
        """
        context.fastmcp_context.set_state(
            FE_CONNECTED_KEY, self.proxy_client.is_connected()
        )
        context.fastmcp_context.set_state(PROXY_TOKEN_KEY, self.proxy_client.wss.token)
        context.fastmcp_context.set_state(PROXY_PORT_KEY, self.proxy_client.wss.port)

        result = await call_next(context)

        connected = self.proxy_client.is_connected()
        connection_state_changed = connected != self.last_message_connected
        self.last_message_connected = connected
        if connection_state_changed:
            await context.fastmcp_context.send_tool_list_changed()

        return result

    async def on_call_tool(self, context, call_next):
        result = await call_next(context)
        if context.message.name != INJECTED_TOOL_NAME:
            return result
        if self.proxy_client.is_connected():
            return result
        # if the tool call was for open_colab_browser_connection and there is no existing connection, try to await full connection
        await context.fastmcp_context.report_progress(
            progress=1, total=3, message="The user is not connected to the Colab UI"
        )
        await context.fastmcp_context.report_progress(
            progress=2,
            total=3,
            message="Waiting for user to connect in Colab - will wait for 60s",
        )

        # PATCH: emit periodic progress while we wait. await_proxy_connection can
        # block for up to UI_CONNECTION_TIMEOUT (60s) with no output, which
        # exceeds the idle timeout of clients like Claude Code (~30s) and causes
        # them to abort the call even though the connection may still succeed
        # server-side. A heartbeat keeps the client's idle timer alive.
        async def _connection_wait_heartbeat():
            elapsed = 0
            try:
                while True:
                    await asyncio.sleep(10)
                    elapsed += 10
                    await context.fastmcp_context.report_progress(
                        progress=2,
                        total=3,
                        message=(
                            f"Still waiting for the Colab UI to connect "
                            f"({elapsed}s elapsed, up to {int(UI_CONNECTION_TIMEOUT)}s)..."
                        ),
                    )
            except asyncio.CancelledError:
                pass

        heartbeat = asyncio.create_task(_connection_wait_heartbeat())
        try:
            await self.proxy_client.await_proxy_connection()
        finally:
            heartbeat.cancel()
        if self.proxy_client.is_connected():
            await context.fastmcp_context.report_progress(
                progress=3, total=3, message="The Colab UI is successfully connected!"
            )
            return ToolResult(
                content=[TextContent(type="text", text="true")],
                structured_content={"result": True},
            )
        else:
            await context.fastmcp_context.report_progress(
                progress=3,
                total=3,
                message="Timeout while waiting for the user to connect.",
            )
            return ToolResult(
                content=[TextContent(type="text", text="false")],
                structured_content={"result": False},
            )


class ColabSessionProxy:
    def __init__(self, browser: ColabBrowser | None = None):
        self._exit_stack = AsyncExitStack()
        self.proxy_server: FastMCPProxy | None = None
        # list order matters, see: https://gofastmcp.com/servers/middleware#multiple-middleware
        self.middleware: list[Middleware] = []
        self.wss: ColabWebSocketServer | None = None
        # A single Playwright-controlled Colab browser, shared by the connection
        # tool (to open + authorize the session, reusing the user's Google
        # login) and by the runtime/screenshot tools.
        self.browser = browser if browser is not None else ColabBrowser()
        self._browser_task: asyncio.Task | None = None
        # Set in start_proxy_server; open_colab_notebook waits on it directly
        # rather than going through ColabProxyMiddleware's connection retry.
        self.proxy_client: ColabProxyClient | None = None
        # Kernel-level tools built on the frontend's proxied notebook tools.
        # It reads self.proxy_client lazily, since that only exists once the
        # proxy server has started.
        self.kernel = KernelTools(lambda: self.proxy_client, browser=self.browser)

    def open_browser(self, url: str) -> None:
        """Launch + authorize the browser off the calling path so we never
        block the event loop; on_call_tool then waits for the frontend to
        connect."""
        self._browser_task = asyncio.create_task(
            self._open_and_connect_browser(url)
        )
        self._browser_task.add_done_callback(_log_task_exception)

    async def _open_and_connect_browser(self, url: str) -> None:
        """Launch the Playwright-controlled Colab browser and click "Connect to
        Colab". Reuses the user's existing Google login, so no re-login is
        needed. Falls back to a non-blocking webbrowser launch if Playwright is
        unavailable.
        """
        try:
            await self.browser.launch(url)
            # Colab bounces to the Google sign-in flow when the copied cookies
            # have expired; click through it before waiting on Colab's own UI.
            await self.browser.ensure_signed_in()
            # The "Connect to Colab" authorization affordance can take a moment
            # to render; poll briefly and click it once it appears.
            for _ in range(15):
                await asyncio.sleep(1)
                if await self.browser.click_connect():
                    break
        except Exception:
            logging.exception(
                "colab-mcp: Playwright launch failed; falling back to webbrowser"
            )
            # webbrowser.open_new() can block on GenericBrowser (it p.wait()s on
            # the child), so run it off the event loop in a detached daemon
            # thread.
            threading.Thread(
                target=webbrowser.open_new, args=(url,), daemon=True
            ).start()

    async def open_notebook(self, url: str, on_progress=None) -> bool:
        """Navigate to `url`, accept the authorization dialog, and wait for the
        Colab frontend to (re)connect to the bridge.

        Returns whether the frontend is connected once we stop waiting. Unlike
        `open_browser` this runs inline, so the caller can report the real
        outcome instead of a fire-and-forget "probably".
        """
        proxy = self.proxy_client
        was_connected = proxy is not None and proxy.is_connected()

        await self.browser.navigate(url)
        await self.browser.ensure_signed_in()

        # Navigating tears down the old tab's websocket, but not instantly: a
        # still-live connection right after the goto belongs to the page we just
        # left, so waiting on it would report success for the wrong notebook.
        if was_connected:
            for _ in range(int(STALE_DISCONNECT_TIMEOUT * 2)):
                if not proxy.is_connected():
                    break
                await asyncio.sleep(0.5)

        elapsed = 0
        while elapsed < UI_CONNECTION_TIMEOUT:
            await asyncio.sleep(1)
            elapsed += 1
            # Safe to retry: click_connect is a no-op unless the authorization
            # dialog is actually on screen.
            await self.browser.click_connect()
            if proxy is not None and proxy.is_connected():
                return True
            if on_progress and elapsed % 10 == 0:
                await on_progress(elapsed)
        return proxy is not None and proxy.is_connected()

    async def import_notebook(self, src_path: str, token, port) -> dict:
        """Upload a local .ipynb into Drive and reconnect the bridge to it.

        The inverse of `export_notebook`. Uploading navigates the tab, which
        drops the websocket, so the notebook is always reopened with a tokened
        URL afterwards — leaving the session pointed at a bridged notebook
        rather than a live tab the editing tools cannot reach.
        """
        if not os.path.isfile(src_path):
            return {"error": f"no such file: {src_path}"}
        # Fail on a malformed notebook here rather than after a browser upload
        # has already put it in the user's Drive.
        try:
            with open(src_path, encoding="utf-8") as handle:
                notebook = json.load(handle)
        except (OSError, ValueError) as e:
            return {"error": f"{src_path} is not readable JSON: {e}"}
        if not isinstance(notebook, dict) or "cells" not in notebook:
            return {"error": f"{src_path} does not look like a .ipynb (no 'cells')"}

        notebook_id = await self.browser.upload_notebook(src_path)
        # Build the Drive path directly: the id came from the URL Colab landed
        # on, so it must not go back through colab_notebook_path's guesswork,
        # which rejects anything under 15 characters as a likely typo.
        url = colab_notebook_url(f"/drive/{notebook_id}", token, port)
        connected = await self.open_notebook(url)
        return {
            "notebook_id": notebook_id,
            "url": url.split("#")[0],
            "cells": len(notebook.get("cells") or []),
            "connected": connected,
        }

    def _make_injected_tools(self) -> list[Tool]:
        async def check_session_proxy_tool_fn(ctx: Context = CurrentContext()) -> bool:
            fe_connected = ctx.get_state(FE_CONNECTED_KEY)
            token = ctx.get_state(PROXY_TOKEN_KEY)
            port = ctx.get_state(PROXY_PORT_KEY)
            if fe_connected:
                return True
            url = f"{COLAB}{SCRATCH_PATH}#mcpProxyToken={token}&mcpProxyPort={port}"
            self.open_browser(url)
            return False

        async def open_colab_notebook_fn(
            notebook: str = "", ctx: Context = CurrentContext()
        ) -> str:
            token = ctx.get_state(PROXY_TOKEN_KEY)
            port = ctx.get_state(PROXY_PORT_KEY)
            try:
                url = colab_notebook_url(notebook, token, port)
            except ValueError as e:
                return f"error: {e}"

            async def on_progress(elapsed: int) -> None:
                await ctx.report_progress(
                    progress=elapsed,
                    total=int(UI_CONNECTION_TIMEOUT),
                    message=(
                        f"Waiting for the Colab UI to connect ({elapsed}s "
                        f"elapsed, up to {int(UI_CONNECTION_TIMEOUT)}s)..."
                    ),
                )

            try:
                connected = await self.open_notebook(url, on_progress=on_progress)
            except Exception as e:  # noqa: BLE001
                logging.exception("colab-mcp: failed to open notebook %s", url)
                return f"error: could not open {url}: {e}"

            # The notebook editing tools are proxied from the frontend, so the
            # client's cached list belongs to the tab we just navigated away
            # from. Ask for a refresh even when connectivity did not flip, which
            # is the case when one notebook is opened straight from another.
            with contextlib.suppress(Exception):
                await ctx.send_tool_list_changed()

            title = await self.browser.title()
            if connected:
                return (
                    f"opened {title or 'notebook'} ({url}); the Colab frontend "
                    "is connected and the notebook editing tools are available"
                )
            return (
                f"opened {title or 'notebook'} ({url}), but the Colab frontend "
                "did not connect within "
                f"{int(UI_CONNECTION_TIMEOUT)}s — the notebook editing tools are "
                "unavailable. Ask the user to accept the 'Connect to a local "
                "Colab MCP server' dialog in the browser window."
            )

        async def set_gpu_runtime_fn(accelerator: str = "T4 GPU") -> str:
            """Change the Colab runtime hardware accelerator via the browser UI."""
            return await self.browser.set_runtime_accelerator(accelerator)

        async def screenshot_colab_fn() -> str:
            """Save a screenshot of the current Colab browser page; returns its path."""
            path = os.path.join(tempfile.gettempdir(), "colab-mcp-screenshot.png")
            return await self.browser.screenshot(path)

        # --- kernel tools -------------------------------------------------
        # Every one of these can be called before a notebook is open, so they
        # report a disconnected frontend as data rather than raising an MCP
        # error the agent then has to interpret.

        async def guarded(coro):
            try:
                return await coro
            except NotConnectedError as e:
                return {"error": str(e), "state": "disconnected"}
            except Exception as e:  # noqa: BLE001
                logging.exception("colab-mcp: kernel tool failed")
                return {"error": f"{type(e).__name__}: {e}"}

        k = self.kernel

        async def interrupt_execution_fn(cancel_pending: bool = True) -> dict:
            return await guarded(k.interrupt(cancel_pending=cancel_pending))

        async def restart_kernel_fn() -> dict:
            return await guarded(k.restart_kernel())

        async def delete_runtime_fn() -> dict:
            return await guarded(k.delete_runtime())

        async def run_code_cell_async_fn(cellId: str) -> dict:
            return await guarded(k.run_cell_async(cellId))

        async def get_execution_status_fn(handle: str, tail_lines: int = 40) -> dict:
            return await guarded(k.execution_status(handle, tail_lines=tail_lines))

        async def get_kernel_status_fn() -> dict:
            return await guarded(k.kernel_status())

        async def execute_transient_fn(code: str, timeout_s: float = 120.0) -> dict:
            return await guarded(k.execute_transient(code, timeout_s=timeout_s))

        async def eval_expression_fn(expr: str, max_repr_chars: int = 4000) -> dict:
            return await guarded(k.eval_expression(expr, max_repr_chars=max_repr_chars))

        async def run_cells_fn(cellIds: list[str], stop_on_error: bool = True) -> list:
            return await guarded(k.run_cells(cellIds, stop_on_error=stop_on_error))

        async def run_all_below_fn(cellId: str, stop_on_error: bool = True) -> list:
            return await guarded(k.run_all_below(cellId, stop_on_error=stop_on_error))

        async def get_cell_output_fn(
            cellId: str, tail_lines: int = 50, max_chars: int = 8000
        ) -> dict:
            return await guarded(
                k.get_cell_output(cellId, tail_lines=tail_lines, max_chars=max_chars)
            )

        async def get_notebook_outline_fn(source_chars: int = 120) -> list:
            return await guarded(k.notebook_outline(source_chars=source_chars))

        async def get_runtime_info_fn() -> dict:
            return await guarded(k.runtime_info())

        async def export_notebook_fn(
            dest_path: str, include_outputs: bool = True
        ) -> dict:
            return await guarded(
                k.export_notebook(dest_path, include_outputs=include_outputs)
            )

        async def download_file_fn(remote_path: str, dest_path: str) -> dict:
            return await guarded(k.download_file(remote_path, dest_path))

        async def upload_file_fn(src_path: str, remote_path: str) -> dict:
            return await guarded(k.upload_file(src_path, remote_path))

        async def import_notebook_fn(
            src_path: str, ctx: Context = CurrentContext()
        ) -> dict:
            token = ctx.get_state(PROXY_TOKEN_KEY)
            port = ctx.get_state(PROXY_PORT_KEY)
            result = await guarded(self.import_notebook(src_path, token, port))
            # The editing tools belong to the tab we just navigated away from,
            # so the client's cached list is stale either way.
            with contextlib.suppress(Exception):
                await ctx.send_tool_list_changed()
            return result

        kernel_tools = [
            Tool.from_function(
                fn=interrupt_execution_fn,
                name="interrupt_execution",
                description=(
                    "Interrupts the cell the Colab kernel is currently running "
                    "(a SIGINT via Runtime > Interrupt execution). This is NOT a "
                    "restart: variables already assigned, including loaded "
                    "models, survive. Also cancels cell runs queued through "
                    "run_code_cell_async unless cancel_pending=false. Returns "
                    "{interrupted, was_running, cancelled_pending}."
                ),
            ),
            Tool.from_function(
                fn=restart_kernel_fn,
                name="restart_kernel",
                description=(
                    "Restarts the Colab kernel (Runtime > Restart session) without "
                    "changing the hardware accelerator — unlike set_gpu_runtime. "
                    "All kernel state is lost. Returns {restarted}."
                ),
            ),
            Tool.from_function(
                fn=delete_runtime_fn,
                name="delete_runtime",
                description=(
                    "Disconnects and DELETES the Colab runtime VM (Runtime > "
                    "Disconnect and delete runtime). Destructive and not "
                    "reversible: unlike restart_kernel, which restarts Python on "
                    "the same machine and leaves /content intact, this discards "
                    "the machine itself — kernel state, uploaded files, pip "
                    "installs and mounted Drive all go with it. Use it to stop an "
                    "idle session consuming compute units, or to force a clean "
                    "machine; use restart_kernel if you only need to clear memory. "
                    "Ask the user before calling this on a session holding work "
                    "they have not saved. Returns {deleted, cancelled_pending, note}."
                ),
            ),
            Tool.from_function(
                fn=run_code_cell_async_fn,
                name="run_code_cell_async",
                description=(
                    "Starts a cell running and returns immediately with an execId "
                    "instead of blocking until the cell finishes. Use this for any "
                    "cell that may take longer than a minute (model loads, sweeps, "
                    "training) and poll get_execution_status; a long cell is then "
                    "reported as 'running' rather than timing out as a failure. "
                    "Returns {execId, cellId, state}."
                ),
            ),
            Tool.from_function(
                fn=get_execution_status_fn,
                name="get_execution_status",
                description=(
                    "Reports on a run started by run_code_cell_async. Accepts an "
                    "execId or a cellId. Returns {state: queued|running|done|error|"
                    "interrupted, elapsed_s, output_tail, display_data, cell_error?}. "
                    "output_tail is the tail of the cell's stdout, so polling a "
                    "long-running cell stays cheap."
                ),
            ),
            Tool.from_function(
                fn=get_kernel_status_fn,
                name="get_kernel_status",
                description=(
                    "Whether the Colab kernel is idle, busy, or disconnected, and "
                    "how many runs are queued — i.e. whether a cell submitted now "
                    "will start immediately or sit behind a long job. Returns "
                    "{state, queue_depth, current_cellId?, elapsed_s?, tracked}. "
                    "tracked=false means the busy cell was detected from the "
                    "frontend rather than started through this server."
                ),
            ),
            Tool.from_function(
                fn=execute_transient_fn,
                name="execute_transient",
                description=(
                    "Runs code in the Colab kernel WITHOUT leaving a cell behind in "
                    "the notebook — the cell is added, run, read, and deleted. Use "
                    "this for debugging one-offs (nvidia-smi, memory probes, shape "
                    "checks) that should not pollute a notebook meant to be shared. "
                    "Returns {stdout, stderr, error, timed_out}."
                ),
            ),
            Tool.from_function(
                fn=eval_expression_fn,
                name="eval_expression",
                description=(
                    "Evaluates a single Python expression in the Colab kernel and "
                    "returns {repr, type, shape?, dtype?, len?, truncated}. Read-only "
                    "(an expression, not a statement) and the repr is truncated in "
                    "the kernel, so inspecting a variable, a tensor shape, or GPU "
                    "memory no longer means editing a cell to add a print and "
                    "re-running it. Leaves no cell behind."
                ),
            ),
            Tool.from_function(
                fn=run_cells_fn,
                name="run_cells",
                description=(
                    "Runs the given cells in order, stopping at the first error "
                    "unless stop_on_error=false. Rebuilding kernel state after a "
                    "runtime switch takes one call instead of one per cell. Returns "
                    "a list of {cellId, state, error?, output_tail}."
                ),
            ),
            Tool.from_function(
                fn=run_all_below_fn,
                name="run_all_below",
                description=(
                    "Runs the given cell and every cell after it, in order. Same "
                    "reporting as run_cells."
                ),
            ),
            Tool.from_function(
                fn=get_cell_output_fn,
                name="get_cell_output",
                description=(
                    "Returns the tail of one cell's output: {stream, display_data, "
                    "error, display_data_current, running}. Prefer this over "
                    "get_cells(includeOutputs=true) when polling — it returns only "
                    "the tail of one cell rather than every cell's full source and "
                    "accumulated output. display_data_current=false means the rich "
                    "output (progress bars in particular) is a stale snapshot and "
                    "should not be trusted; the 'stream' text is always current."
                ),
            ),
            Tool.from_function(
                fn=get_notebook_outline_fn,
                name="get_notebook_outline",
                description=(
                    "A cheap map of the notebook: per cell, {index, cellId, "
                    "source_head, source_lines, output_chars, has_error}. Use this "
                    "instead of get_cells when you only need to find a cell, since "
                    "it returns no full sources and no outputs."
                ),
            ),
            Tool.from_function(
                fn=get_runtime_info_fn,
                name="get_runtime_info",
                description=(
                    "Hardware and resource state of the Colab runtime: accelerator, "
                    "gpu_name, vram_total_mb, vram_used_mb, ram_total_mb, "
                    "disk_free_gb, connected. Runs transiently, so it leaves no cell "
                    "behind — no need to add an nvidia-smi cell to the notebook."
                ),
            ),
            Tool.from_function(
                fn=export_notebook_fn,
                name="export_notebook",
                description=(
                    "Saves the notebook and downloads the .ipynb to a local path via "
                    "the browser, so the file goes straight to disk and costs nothing "
                    "in context. Pass include_outputs=false to strip cell outputs. "
                    "Returns {path, bytes, cells, saved_before_export}."
                ),
            ),
            Tool.from_function(
                fn=download_file_fn,
                name="download_file",
                description=(
                    "Copies a file from the Colab VM's filesystem to a local path. "
                    "Transferred in chunks through the kernel, so the bytes never "
                    "enter the conversation. Returns {path, bytes, remote_bytes}."
                ),
            ),
            Tool.from_function(
                fn=upload_file_fn,
                name="upload_file",
                description=(
                    "Copies a local file onto the Colab VM's filesystem (e.g. "
                    "/content/data.pt), in chunks through the kernel so the bytes "
                    "never enter the conversation. Note the VM's disk is scratch "
                    "space and does NOT survive restart_kernel or a runtime "
                    "change; to upload a notebook use import_notebook instead. "
                    "Returns {remote_path, bytes}."
                ),
            ),
            Tool.from_function(
                fn=import_notebook_fn,
                name="import_notebook",
                description=(
                    "Uploads a local .ipynb into Google Drive as a real Colab "
                    "notebook and opens it — the inverse of export_notebook, and "
                    "unlike upload_file the result is a notebook you can edit and "
                    "run rather than bytes on the VM's scratch disk. This SWITCHES "
                    "the active notebook: the imported one is reopened with the "
                    "bridge attached, so the editing tools then apply to it. "
                    "Returns {notebook_id, url, cells, connected}."
                ),
            ),
        ]

        return kernel_tools + [
            Tool.from_function(
                fn=check_session_proxy_tool_fn,
                name=INJECTED_TOOL_NAME,
                description=(
                    "Opens a connection to a Google Colab browser session and "
                    "unlocks notebook editing tools. Returns a boolean "
                    "representing whether the connection attempt succeeded"
                ),
            ),
            Tool.from_function(
                fn=open_colab_notebook_fn,
                name=OPEN_NOTEBOOK_TOOL_NAME,
                description=(
                    "Opens a specific notebook in the Google Colab browser session "
                    "and unlocks the notebook editing tools for it. Accepts a Colab "
                    "URL, a Google Drive file id or Drive URL, a GitHub blob or gist "
                    "URL, or a Colab path; omit the argument (or pass 'new') for a "
                    "blank scratch notebook. Launches the browser if it is not "
                    "already open, so this can be called instead of "
                    "open_colab_browser_connection. Returns a status string."
                ),
            ),
            Tool.from_function(
                fn=set_gpu_runtime_fn,
                name="set_gpu_runtime",
                description=(
                    "Changes the Google Colab runtime hardware accelerator via the Colab UI "
                    "(Runtime > Change runtime type). Pass one of: 'T4 GPU', 'L4 GPU', "
                    "'A100 GPU', 'G4 GPU', 'v6e-1 TPU', 'v5e-1 TPU', or 'CPU'. Requires the "
                    "browser opened by open_colab_browser_connection. Returns a status string."
                ),
            ),
            Tool.from_function(
                fn=screenshot_colab_fn,
                name="screenshot_colab",
                description=(
                    "Saves a PNG screenshot of the current Colab browser page (the "
                    "Playwright-controlled window) and returns the file path."
                ),
            ),
        ]

    async def start_proxy_server(self):
        self._exit_stack.push_async_callback(self.browser.close)
        self.wss = await self._exit_stack.enter_async_context(ColabWebSocketServer())
        proxy_client = await self._exit_stack.enter_async_context(
            ColabProxyClient(self.wss)
        )
        self.proxy_client = proxy_client
        self.proxy_server = FastMCPProxy(
            client_factory=proxy_client.client_factory,
            instructions="Connects to a user's Google Colab session in a browser and allows for interactions with their Google Colab notebook",
        )
        # ColabProxyMiddleware must be first because it sets the fe_connected state
        self.middleware.append(ColabProxyMiddleware(proxy_client))
        self.middleware.append(
            ToolInjectionMiddleware(tools=self._make_injected_tools())
        )

    async def cleanup(self):
        await self._exit_stack.aclose()
