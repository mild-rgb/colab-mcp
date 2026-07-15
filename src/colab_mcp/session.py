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
import logging
import os
import tempfile
import threading
import webbrowser

from colab_mcp.browser import ColabBrowser
from colab_mcp.websocket_server import ColabWebSocketServer, COLAB, SCRATCH_PATH

UI_CONNECTION_TIMEOUT = 60.0  # secs

FE_CONNECTED_KEY = "fe_connected"
PROXY_TOKEN_KEY = "proxy_token"
PROXY_PORT_KEY = "proxy_port"
INJECTED_TOOL_NAME = "open_colab_browser_connection"


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

        async def set_gpu_runtime_fn(accelerator: str = "T4 GPU") -> str:
            """Change the Colab runtime hardware accelerator via the browser UI."""
            return await self.browser.set_runtime_accelerator(accelerator)

        async def screenshot_colab_fn() -> str:
            """Save a screenshot of the current Colab browser page; returns its path."""
            path = os.path.join(tempfile.gettempdir(), "colab-mcp-screenshot.png")
            return await self.browser.screenshot(path)

        return [
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
