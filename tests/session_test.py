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
import json
from colab_mcp import session
from colab_mcp.websocket_server import ColabWebSocketServer
from fastmcp.server.middleware import MiddlewareContext
import pytest
from unittest.mock import patch, AsyncMock, Mock
import websockets


@pytest.fixture(autouse=True)
def mock_webbrowser(monkeypatch):
    import webbrowser

    mock_open = Mock()
    monkeypatch.setattr(webbrowser, "open_new", mock_open)
    return mock_open


@pytest.fixture
def mock_wss():
    """Provides a mock ColabWebSocketServer instance."""
    return MockColabWebSocketServer()


class MockColabWebSocketServer:
    def __init__(self):
        self.connection_live = asyncio.Event()
        self.connection_closed = asyncio.Event()
        self.read_stream = AsyncMock()
        self.write_stream = AsyncMock()
        self.token = "test-token"
        self.port = 1234

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def mock_proxy_client(mock_wss):
    client = Mock(spec=session.ColabProxyClient)
    client.wss = mock_wss
    client.is_connected.return_value = False
    return client


class TestColabProxyMiddleware:
    @pytest.mark.asyncio
    async def test_connection_live(self, mock_proxy_client):
        """Tests connection state change from disconnected to connected."""
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = True
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", True)
        context.fastmcp_context.set_state.assert_any_call("proxy_token", "test-token")
        context.fastmcp_context.set_state.assert_any_call("proxy_port", 1234)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_not_live(self, mock_proxy_client):
        """Tests connection state change from connected to disconnected."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        mock_proxy_client.is_connected.return_value = False
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", False)
        assert middleware.last_message_connected is False
        context.fastmcp_context.send_tool_list_changed.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_connection_change(self, mock_proxy_client):
        """Tests no connection state change."""
        mock_proxy_client.is_connected.return_value = True
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock(spec=MiddlewareContext)
        context.fastmcp_context.set_state = Mock()
        context.fastmcp_context.send_tool_list_changed = AsyncMock()
        call_next = AsyncMock()

        await middleware.on_message(context, call_next)

        call_next.assert_called_once_with(context)
        context.fastmcp_context.set_state.assert_any_call("fe_connected", True)
        assert middleware.last_message_connected is True
        context.fastmcp_context.send_tool_list_changed.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_call_tool_await_connection(self, mock_proxy_client):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.side_effect = [False, True]
        mock_proxy_client.await_proxy_connection = AsyncMock()
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        mock_proxy_client.await_proxy_connection.assert_called_once()
        context.fastmcp_context.report_progress.assert_called()
        assert result.structured_content == {"result": True}

    @pytest.mark.asyncio
    async def test_on_call_tool_timeout(self, mock_proxy_client):
        middleware = session.ColabProxyMiddleware(mock_proxy_client)
        context = Mock()
        context.fastmcp_context.report_progress = AsyncMock()
        context.message.name = session.INJECTED_TOOL_NAME
        mock_proxy_client.is_connected.return_value = False
        mock_proxy_client.await_proxy_connection = AsyncMock()
        call_next = AsyncMock()

        result = await middleware.on_call_tool(context, call_next)

        mock_proxy_client.await_proxy_connection.assert_called_once()
        assert result.structured_content == {"result": False}


def _get_injected_tool_fn(proxy, name):
    tool = next(t for t in proxy._make_injected_tools() if t.name == name)
    return tool.fn


class TestCheckSessionProxyToolFn:
    @pytest.mark.asyncio
    async def test_connected(self):
        proxy = session.ColabSessionProxy(browser=Mock())
        ctx = Mock()
        ctx.get_state.side_effect = (
            lambda k: True if k == session.FE_CONNECTED_KEY else None
        )
        tool_fn = _get_injected_tool_fn(proxy, session.INJECTED_TOOL_NAME)
        assert await tool_fn(ctx) is True

    @pytest.mark.asyncio
    async def test_disconnected(self):
        proxy = session.ColabSessionProxy(browser=Mock())
        proxy.open_browser = Mock()
        ctx = Mock()

        def get_state(k):
            if k == session.FE_CONNECTED_KEY:
                return False
            if k == session.PROXY_TOKEN_KEY:
                return "test-token"
            if k == session.PROXY_PORT_KEY:
                return 1234
            return None

        ctx.get_state.side_effect = get_state
        tool_fn = _get_injected_tool_fn(proxy, session.INJECTED_TOOL_NAME)
        assert await tool_fn(ctx) is False
        proxy.open_browser.assert_called_once()
        (url,), _ = proxy.open_browser.call_args
        assert "mcpProxyToken=test-token" in url
        assert "mcpProxyPort=1234" in url

    @pytest.mark.asyncio
    async def test_open_browser_webbrowser_fallback(self, mock_webbrowser):
        """open_browser keeps a task reference and falls back to webbrowser
        when the Playwright launch fails."""
        browser = Mock()
        browser.launch = AsyncMock(side_effect=RuntimeError("no playwright"))
        proxy = session.ColabSessionProxy(browser=browser)

        proxy.open_browser("http://example.test/nb")

        assert isinstance(proxy._browser_task, asyncio.Task)
        await asyncio.wait_for(proxy._browser_task, timeout=2)
        # the fallback runs webbrowser.open_new in a short-lived thread
        for _ in range(100):
            if mock_webbrowser.call_count:
                break
            await asyncio.sleep(0.01)
        mock_webbrowser.assert_called_once_with("http://example.test/nb")


class TestColabProxyClient:
    def test_is_connected(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.is_connected() is False
        mock_wss.connection_live.set()
        assert client.is_connected() is False
        client.proxy_mcp_client = Mock()
        assert client.is_connected() is True

    def test_client_factory_connection_live(self, mock_wss):
        mock_wss.connection_live.set()
        client = session.ColabProxyClient(mock_wss)
        client.proxy_mcp_client = Mock()

        assert client.client_factory() is client.proxy_mcp_client

    def test_client_factory_connection_not_live(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        assert client.client_factory() is client.stubbed_mcp_client

    @pytest.mark.asyncio
    async def test_await_proxy_connection_timeout(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        with patch("colab_mcp.session.UI_CONNECTION_TIMEOUT", 0.05):
            # returns quietly on timeout instead of raising
            await client.await_proxy_connection()

    @pytest.mark.asyncio
    async def test_await_proxy_connection_ready(self, mock_wss):
        client = session.ColabProxyClient(mock_wss)
        client.client_ready.set()
        await asyncio.wait_for(client.await_proxy_connection(), timeout=1)

    @pytest.mark.asyncio
    @patch("colab_mcp.session.Client")
    @patch("colab_mcp.session.ColabTransport", spec=session.ColabTransport)
    async def test_manager_connects_and_reconnects(
        self, mock_colab_transport, mock_client, mock_wss
    ):
        mock_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_client.return_value
        )
        mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
        async with session.ColabProxyClient(mock_wss) as client:
            # frontend connects
            mock_wss.connection_closed.clear()
            mock_wss.connection_live.set()
            await asyncio.wait_for(client.client_ready.wait(), timeout=1)
            assert client.is_connected()
            mock_colab_transport.assert_called_with(mock_wss)

            # frontend disconnects: the proxy client is torn down
            mock_wss.connection_live.clear()
            mock_wss.connection_closed.set()
            for _ in range(100):
                if not client.client_ready.is_set():
                    break
                await asyncio.sleep(0.01)
            assert client.proxy_mcp_client is None
            assert not client.is_connected()

            # frontend reconnects: a fresh client session is established
            mock_wss.connection_closed.clear()
            mock_wss.connection_live.set()
            await asyncio.wait_for(client.client_ready.wait(), timeout=1)
            assert client.is_connected()
            # one Client per frontend connection (the stub client in __init__
            # also counts one Client() call, so filter to transport-backed ones)
            proxy_session_calls = [
                c
                for c in mock_client.call_args_list
                if c.args and c.args[0] is mock_colab_transport.return_value
            ]
            assert len(proxy_session_calls) == 2


async def _serve_fake_frontend_handshake(ws):
    """Act as the Colab frontend's MCP server for the initialize handshake."""
    init = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    assert init["method"] == "initialize"
    await ws.send(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": init["id"],
                "result": {
                    "protocolVersion": init["params"]["protocolVersion"],
                    "capabilities": {},
                    "serverInfo": {"name": "fake-colab", "version": "0.0"},
                },
            }
        )
    )
    note = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
    assert note["method"] == "notifications/initialized"


class TestFrontendReconnect:
    @pytest.mark.asyncio
    async def test_reconnect_gets_fresh_session(self):
        """A Colab tab that disconnects and reconnects must be able to complete
        a brand-new MCP handshake (regression: the old single-shot client left
        the bridge unusable after the first disconnect)."""
        async with ColabWebSocketServer() as wss:
            async with session.ColabProxyClient(wss) as pc:
                for _attempt in range(2):
                    ws = await websockets.connect(
                        f"ws://localhost:{wss.port}?access_token={wss.token}",
                        origin="https://colab.google.com",
                        subprotocols=["mcp"],
                    )
                    await _serve_fake_frontend_handshake(ws)
                    await asyncio.wait_for(pc.client_ready.wait(), timeout=5)
                    assert pc.is_connected()

                    await ws.close()
                    for _ in range(500):
                        if (
                            not pc.client_ready.is_set()
                            and not wss.connection_live.is_set()
                        ):
                            break
                        await asyncio.sleep(0.01)
                    assert not pc.is_connected()


class TestColabTransport:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ClientSession")
    async def test_connect_session(self, mock_client_session, mock_wss):
        transport = session.ColabTransport(mock_wss)
        mock_client_session.return_value.__aenter__ = AsyncMock()
        async with transport.connect_session(foo="bar") as client_session:
            assert (
                client_session
                == mock_client_session.return_value.__aenter__.return_value
            )

        mock_client_session.assert_called_once_with(
            mock_wss.read_stream, mock_wss.write_stream, foo="bar"
        )


class TestColabSessionProxy:
    @pytest.mark.asyncio
    @patch("colab_mcp.session.ToolInjectionMiddleware")
    @patch("colab_mcp.session.ColabWebSocketServer")
    @patch("colab_mcp.session.ColabProxyClient")
    @patch("colab_mcp.session.ColabProxyMiddleware")
    async def test_start_proxy_server(
        self,
        mock_colab_proxy_middleware,
        mock_colab_proxy_client,
        mock_colab_web_socket_server,
        mock_tool_injection_middleware,
    ):
        mock_colab_web_socket_server.return_value.__aenter__ = AsyncMock()
        mock_colab_proxy_client.return_value.__aenter__ = AsyncMock()
        proxy = session.ColabSessionProxy()
        await proxy.start_proxy_server()
        mock_colab_proxy_client.assert_called_once()
        assert proxy.proxy_server is not None
        mock_colab_proxy_middleware.assert_called_once()
        mock_tool_injection_middleware.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        proxy = session.ColabSessionProxy()
        proxy._exit_stack = AsyncMock()
        await proxy.cleanup()
        proxy._exit_stack.aclose.assert_called_once()


class TestColabNotebookPath:
    @pytest.mark.parametrize(
        "ref,expected",
        [
            ("", "/notebooks/empty.ipynb"),
            ("new", "/notebooks/empty.ipynb"),
            ("  Scratch  ", "/notebooks/empty.ipynb"),
            (
                "1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy",
                "/drive/1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy",
            ),
            ("/drive/abc123", "/drive/abc123"),
            (
                "https://colab.research.google.com/drive/1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy",
                "/drive/1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy",
            ),
            (
                "colab.research.google.com/drive/abcdefghijklmnop",
                "/drive/abcdefghijklmnop",
            ),
            (
                "https://drive.google.com/file/d/1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy/view?usp=sharing",
                "/drive/1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy",
            ),
            (
                "https://github.com/owner/repo/blob/main/nb.ipynb",
                "/github/owner/repo/blob/main/nb.ipynb",
            ),
            (
                "https://gist.github.com/owner/deadbeef",
                "/gist/owner/deadbeef",
            ),
        ],
    )
    def test_resolves(self, ref, expected):
        assert session.colab_notebook_path(ref) == expected

    def test_drive_url_without_id_is_rejected(self):
        with pytest.raises(ValueError, match="no Drive file id"):
            session.colab_notebook_path("https://drive.google.com/drive/my-drive")

    def test_unknown_host_is_rejected(self):
        with pytest.raises(ValueError, match="unrecognised notebook reference"):
            session.colab_notebook_path("https://example.test/nb.ipynb")

    def test_url_carries_current_token_and_drops_stale_fragment(self):
        url = session.colab_notebook_url(
            "https://colab.research.google.com/drive/abc123def456ghi"
            "#mcpProxyToken=STALE&mcpProxyPort=1",
            "fresh-token",
            4321,
        )
        assert url == (
            "https://colab.research.google.com/drive/abc123def456ghi"
            "#mcpProxyToken=fresh-token&mcpProxyPort=4321"
        )
        assert "STALE" not in url


class TestOpenNotebook:
    def _proxy(self, connected_after=0):
        """Proxy whose frontend reports connected after N is_connected calls."""
        browser = Mock()
        browser.navigate = AsyncMock()
        browser.ensure_signed_in = AsyncMock(return_value=True)
        browser.click_connect = AsyncMock(return_value=False)
        browser.title = AsyncMock(return_value="nb.ipynb - Colab")
        proxy = session.ColabSessionProxy(browser=browser)
        calls = {"n": 0}

        def is_connected():
            calls["n"] += 1
            return calls["n"] > connected_after

        proxy.proxy_client = Mock(spec=session.ColabProxyClient)
        proxy.proxy_client.is_connected.side_effect = is_connected
        return proxy

    @pytest.mark.asyncio
    async def test_navigates_and_reports_connection(self, monkeypatch):
        monkeypatch.setattr(session, "UI_CONNECTION_TIMEOUT", 3)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        proxy = self._proxy(connected_after=1)  # not connected before navigating

        assert await proxy.open_notebook("http://colab.test/nb#tok") is True
        proxy.browser.navigate.assert_awaited_once_with("http://colab.test/nb#tok")

    @pytest.mark.asyncio
    async def test_waits_for_stale_connection_to_drop(self, monkeypatch):
        """A live socket immediately after navigating belongs to the old tab."""
        monkeypatch.setattr(session, "UI_CONNECTION_TIMEOUT", 3)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        browser = Mock()
        browser.navigate = AsyncMock()
        browser.ensure_signed_in = AsyncMock(return_value=True)
        browser.click_connect = AsyncMock(return_value=False)
        proxy = session.ColabSessionProxy(browser=browser)
        # connected, then the old tab drops, then the new tab connects
        states = [True, True, False, True]
        proxy.proxy_client = Mock(spec=session.ColabProxyClient)
        proxy.proxy_client.is_connected.side_effect = (
            lambda: states.pop(0) if states else True
        )

        assert await proxy.open_notebook("http://colab.test/nb") is True
        assert states == []  # the stale-drop poll consumed the leading Trues

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self, monkeypatch):
        monkeypatch.setattr(session, "UI_CONNECTION_TIMEOUT", 3)
        monkeypatch.setattr(asyncio, "sleep", AsyncMock())
        browser = Mock()
        browser.navigate = AsyncMock()
        browser.ensure_signed_in = AsyncMock(return_value=True)
        browser.click_connect = AsyncMock(return_value=False)
        proxy = session.ColabSessionProxy(browser=browser)
        proxy.proxy_client = Mock(spec=session.ColabProxyClient)
        proxy.proxy_client.is_connected.return_value = False

        assert await proxy.open_notebook("http://colab.test/nb") is False
        # keeps retrying the dialog for the whole window
        assert browser.click_connect.await_count == 3


class TestOpenColabNotebookToolFn:
    def _ctx(self):
        ctx = Mock()
        ctx.get_state.side_effect = lambda k: {
            session.PROXY_TOKEN_KEY: "test-token",
            session.PROXY_PORT_KEY: 1234,
        }.get(k)
        ctx.report_progress = AsyncMock()
        ctx.send_tool_list_changed = AsyncMock()
        return ctx

    @pytest.mark.asyncio
    async def test_opens_and_refreshes_tool_list(self):
        browser = Mock()
        browser.title = AsyncMock(return_value="junk_tokens_steering.ipynb - Colab")
        proxy = session.ColabSessionProxy(browser=browser)
        proxy.open_notebook = AsyncMock(return_value=True)
        ctx = self._ctx()

        tool_fn = _get_injected_tool_fn(proxy, session.OPEN_NOTEBOOK_TOOL_NAME)
        result = await tool_fn("1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy", ctx)

        (url,), kwargs = proxy.open_notebook.call_args
        assert url == (
            "https://colab.research.google.com/drive/1oiVJRXMQfLAIlVsmRcxF5lkYSdBnjtxy"
            "#mcpProxyToken=test-token&mcpProxyPort=1234"
        )
        assert "junk_tokens_steering.ipynb" in result
        assert "connected" in result
        # the proxied editing tools belong to the tab we just left
        ctx.send_tool_list_changed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reports_unconnected_frontend(self):
        browser = Mock()
        browser.title = AsyncMock(return_value="nb.ipynb - Colab")
        proxy = session.ColabSessionProxy(browser=browser)
        proxy.open_notebook = AsyncMock(return_value=False)

        tool_fn = _get_injected_tool_fn(proxy, session.OPEN_NOTEBOOK_TOOL_NAME)
        result = await tool_fn("abc123def456ghi", self._ctx())
        assert "did not connect" in result

    @pytest.mark.asyncio
    async def test_bad_reference_returns_error_without_navigating(self):
        proxy = session.ColabSessionProxy(browser=Mock())
        proxy.open_notebook = AsyncMock()

        tool_fn = _get_injected_tool_fn(proxy, session.OPEN_NOTEBOOK_TOOL_NAME)
        result = await tool_fn("https://example.test/nb.ipynb", self._ctx())
        assert result.startswith("error:")
        proxy.open_notebook.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_navigation_failure_is_reported(self):
        proxy = session.ColabSessionProxy(browser=Mock())
        proxy.open_notebook = AsyncMock(side_effect=RuntimeError("page crashed"))

        tool_fn = _get_injected_tool_fn(proxy, session.OPEN_NOTEBOOK_TOOL_NAME)
        result = await tool_fn("abc123def456ghi", self._ctx())
        assert result.startswith("error: could not open")
        assert "page crashed" in result


class TestKernelToolWiring:
    """The kernel tools are injected alongside — never instead of — the
    browser-level tools the improvements doc asked us not to regress."""

    KERNEL_TOOLS = [
        "interrupt_execution",
        "restart_kernel",
        "run_code_cell_async",
        "get_execution_status",
        "get_kernel_status",
        "execute_transient",
        "eval_expression",
        "run_cells",
        "run_all_below",
        "get_cell_output",
        "get_notebook_outline",
        "get_runtime_info",
        "export_notebook",
        "download_file",
        "upload_file",
    ]

    PRESERVED_TOOLS = [
        session.INJECTED_TOOL_NAME,
        session.OPEN_NOTEBOOK_TOOL_NAME,
        "set_gpu_runtime",
        "screenshot_colab",
    ]

    def _names(self):
        proxy = session.ColabSessionProxy(browser=Mock())
        return [t.name for t in proxy._make_injected_tools()]

    def test_every_kernel_tool_is_injected(self):
        names = self._names()
        assert [t for t in self.KERNEL_TOOLS if t not in names] == []

    def test_the_existing_tools_are_preserved(self):
        names = self._names()
        assert [t for t in self.PRESERVED_TOOLS if t not in names] == []

    def test_tool_names_are_unique(self):
        names = self._names()
        assert len(names) == len(set(names))

    def test_every_tool_has_a_description(self):
        proxy = session.ColabSessionProxy(browser=Mock())
        assert all(t.description for t in proxy._make_injected_tools())

    def test_the_kernel_shares_the_sessions_browser(self):
        browser = Mock()
        proxy = session.ColabSessionProxy(browser=browser)
        assert proxy.kernel.browser is browser

    def test_the_kernel_reads_the_proxy_client_lazily(self):
        """KernelTools is built in __init__, before start_proxy_server exists."""
        proxy = session.ColabSessionProxy(browser=Mock())
        assert proxy.kernel.tools.is_connected() is False
        proxy.proxy_client = Mock()
        proxy.proxy_client.is_connected.return_value = True
        assert proxy.kernel.tools.is_connected() is True

    @pytest.mark.asyncio
    async def test_a_disconnected_frontend_is_reported_as_data(self):
        """These tools are callable before a notebook is open, so a missing
        frontend must not surface as an MCP error the agent has to interpret."""
        proxy = session.ColabSessionProxy(browser=Mock())
        tool_fn = _get_injected_tool_fn(proxy, "get_kernel_status")

        result = await tool_fn()

        assert result["state"] == "disconnected"

    @pytest.mark.asyncio
    async def test_an_unexpected_failure_is_reported_as_data(self):
        proxy = session.ColabSessionProxy(browser=Mock())
        proxy.kernel.eval_expression = AsyncMock(side_effect=RuntimeError("kaboom"))
        tool_fn = _get_injected_tool_fn(proxy, "eval_expression")

        result = await tool_fn("1 + 1")

        assert result == {"error": "RuntimeError: kaboom"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_name,kernel_method,args,expected",
        [
            ("interrupt_execution", "interrupt", (), {"cancel_pending": True}),
            ("restart_kernel", "restart_kernel", (), {}),
            ("run_code_cell_async", "run_cell_async", ("c1",), {}),
            ("get_kernel_status", "kernel_status", (), {}),
            ("get_runtime_info", "runtime_info", (), {}),
            ("run_cells", "run_cells", (["a", "b"],), {"stop_on_error": True}),
            ("run_all_below", "run_all_below", ("c1",), {"stop_on_error": True}),
            ("download_file", "download_file", ("/r", "/l"), {}),
            ("upload_file", "upload_file", ("/l", "/r"), {}),
        ],
    )
    async def test_tools_delegate_to_the_kernel(
        self, tool_name, kernel_method, args, expected
    ):
        proxy = session.ColabSessionProxy(browser=Mock())
        stub = AsyncMock(return_value={"ok": True})
        setattr(proxy.kernel, kernel_method, stub)

        result = await _get_injected_tool_fn(proxy, tool_name)(*args)

        assert result == {"ok": True}
        called_args, called_kwargs = stub.call_args
        assert called_args == args
        for key, value in expected.items():
            assert called_kwargs[key] == value


class TestImportNotebook:
    """import_notebook is the inverse of export_notebook: a local .ipynb becomes
    a real Drive notebook, not bytes on the VM's scratch disk."""

    def _proxy(self, notebook_id="NEWID123", connected=True):
        browser = Mock()
        browser.upload_notebook = AsyncMock(return_value=notebook_id)
        proxy = session.ColabSessionProxy(browser=browser)
        proxy.open_notebook = AsyncMock(return_value=connected)
        return proxy

    def _notebook(self, tmp_path, cells=2):
        path = tmp_path / "local.ipynb"
        path.write_text(
            json.dumps({"cells": [{"cell_type": "code", "source": []}] * cells, "nbformat": 4})
        )
        return str(path)

    @pytest.mark.asyncio
    async def test_uploads_then_reattaches_the_bridge(self, tmp_path):
        """Uploading navigates the tab and drops the websocket, so the imported
        notebook must be reopened with a tokened URL."""
        proxy = self._proxy()
        src = self._notebook(tmp_path)

        result = await proxy.import_notebook(src, "tok", 4321)

        proxy.browser.upload_notebook.assert_awaited_once_with(src)
        (url,), _ = proxy.open_notebook.call_args
        assert "/drive/NEWID123" in url
        assert "mcpProxyToken=tok" in url and "mcpProxyPort=4321" in url
        assert result["notebook_id"] == "NEWID123"
        assert result["cells"] == 2
        assert result["connected"] is True
        assert "#" not in result["url"]

    @pytest.mark.asyncio
    async def test_reports_a_bridge_that_did_not_reconnect(self, tmp_path):
        proxy = self._proxy(connected=False)
        result = await proxy.import_notebook(self._notebook(tmp_path), "tok", 1)
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_rejects_a_missing_file_without_touching_the_browser(self, tmp_path):
        proxy = self._proxy()
        result = await proxy.import_notebook(str(tmp_path / "absent.ipynb"), "tok", 1)
        assert "no such file" in result["error"]
        proxy.browser.upload_notebook.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_unparseable_json_before_uploading(self, tmp_path):
        """Validate locally rather than after it is already in the user's Drive."""
        proxy = self._proxy()
        bad = tmp_path / "bad.ipynb"
        bad.write_text("{not json")

        result = await proxy.import_notebook(str(bad), "tok", 1)

        assert "not readable JSON" in result["error"]
        proxy.browser.upload_notebook.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_rejects_json_that_is_not_a_notebook(self, tmp_path):
        proxy = self._proxy()
        wrong = tmp_path / "wrong.ipynb"
        wrong.write_text(json.dumps({"hello": "world"}))

        result = await proxy.import_notebook(str(wrong), "tok", 1)

        assert "does not look like a .ipynb" in result["error"]
        proxy.browser.upload_notebook.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_tool_is_injected_and_refreshes_the_tool_list(self, tmp_path):
        proxy = self._proxy()
        ctx = Mock()
        ctx.get_state.side_effect = lambda k: {
            session.PROXY_TOKEN_KEY: "tok",
            session.PROXY_PORT_KEY: 4321,
        }.get(k)
        ctx.send_tool_list_changed = AsyncMock()

        tool_fn = _get_injected_tool_fn(proxy, "import_notebook")
        result = await tool_fn(self._notebook(tmp_path), ctx)

        assert result["notebook_id"] == "NEWID123"
        # the editing tools belong to the tab we navigated away from
        ctx.send_tool_list_changed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_upload_failure_is_reported_as_data(self, tmp_path):
        proxy = self._proxy()
        proxy.browser.upload_notebook = AsyncMock(side_effect=RuntimeError("no menu"))
        ctx = Mock()
        ctx.get_state.return_value = "tok"
        ctx.send_tool_list_changed = AsyncMock()

        tool_fn = _get_injected_tool_fn(proxy, "import_notebook")
        result = await tool_fn(self._notebook(tmp_path), ctx)

        assert result == {"error": "RuntimeError: no menu"}
