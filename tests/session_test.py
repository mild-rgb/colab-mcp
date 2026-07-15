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
