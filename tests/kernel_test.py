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

"""Regression tests for the kernel tools.

Most of these are pinned to behaviour observed against a live Colab session
(see tests/fake_frontend.py for the shapes). Where a test guards a defect that
was actually hit, the docstring says so.
"""

import asyncio
import json
import os

import pytest

from colab_mcp import kernel
from colab_mcp.kernel import (
    ExecutionManager,
    KernelTools,
    NotConnectedError,
    ProxyTools,
    adapt_args,
    extract_payload,
    normalize_cells,
    result_payload,
    split_outputs,
    strip_notebook_outputs,
    strip_payload,
    tail,
)
from fake_frontend import (
    FRONTEND_SCHEMAS,
    FakeFrontend,
    FakeProxyClient,
    FakeToolResult,
    display_data,
    error_output,
    execute_result,
    stream_output,
)
from unittest.mock import AsyncMock, Mock

from colab_mcp import browser as kernel_browser


@pytest.fixture
def frontend():
    return FakeFrontend()


@pytest.fixture
def tools(frontend):
    return ProxyTools(FakeProxyClient(frontend))


@pytest.fixture
def browser():
    b = Mock()
    b.interrupt_execution = AsyncMock(return_value=True)
    b.restart_kernel = AsyncMock(return_value=True)
    b.delete_runtime = AsyncMock(return_value=True)
    b.runtime_attached = AsyncMock(return_value=None)
    b.save_notebook = AsyncMock(return_value=True)
    b.download_notebook = AsyncMock()
    return b


@pytest.fixture
def kernel_tools(frontend, browser):
    return KernelTools(FakeProxyClient(frontend), browser=browser)


# --------------------------------------------------------------------------
# argument / tool-name adaptation
# --------------------------------------------------------------------------


class TestAdaptArgs:
    def test_maps_content_to_the_frontends_code_arg(self):
        """The frontend calls it `code`; callers here say `content`."""
        out = adapt_args(FRONTEND_SCHEMAS["add_code_cell"], {"content": "x = 1"})
        assert out == {"code": "x = 1"}

    def test_maps_index_to_cell_index(self):
        out = adapt_args(FRONTEND_SCHEMAS["add_code_cell"], {"index": 3})
        assert out == {"cellIndex": 3}

    def test_maps_range_args(self):
        out = adapt_args(
            FRONTEND_SCHEMAS["get_cells"], {"indexStart": 1, "indexEnd": 2}
        )
        assert out == {"cellIndexStart": 1, "cellIndexEnd": 2}

    def test_passes_through_exact_matches(self):
        out = adapt_args(FRONTEND_SCHEMAS["run_code_cell"], {"cellId": "abc"})
        assert out == {"cellId": "abc"}

    def test_drops_none_values(self):
        """Callers pass optional args unconditionally; None must not be sent."""
        out = adapt_args(
            FRONTEND_SCHEMAS["get_cells"],
            {"includeOutputs": True, "indexStart": None, "indexEnd": None},
        )
        assert out == {"includeOutputs": True}

    def test_drops_args_the_schema_does_not_declare(self):
        out = adapt_args(FRONTEND_SCHEMAS["delete_cell"], {"cellId": "a", "bogus": 1})
        assert out == {"cellId": "a"}

    def test_passes_everything_through_when_schema_is_unknown(self):
        """Better a loud frontend rejection than a silently dropped argument."""
        assert adapt_args(None, {"cellId": "a"}) == {"cellId": "a"}
        assert adapt_args({}, {"cellId": "a"}) == {"cellId": "a"}

    def test_matches_case_and_underscore_variants(self):
        schema = {"properties": {"cell_id": {}}}
        assert adapt_args(schema, {"cellId": "a"}) == {"cell_id": "a"}


class TestToolResolution:
    @pytest.mark.asyncio
    async def test_resolves_the_live_frontend_names(self, tools):
        for canonical in kernel.TOOL_ALIASES:
            assert await tools.resolve_tool(canonical) == canonical

    @pytest.mark.asyncio
    async def test_resolves_an_alternative_spelling(self, frontend):
        frontend_schemas = {"runCodeCell": FRONTEND_SCHEMAS["run_code_cell"]}

        class Renamed(FakeFrontend):
            async def list_tools(self):
                from fake_frontend import FakeToolDef

                return [
                    FakeToolDef(name=n, inputSchema=s)
                    for n, s in frontend_schemas.items()
                ]

        tools = ProxyTools(FakeProxyClient(Renamed()))
        assert await tools.resolve_tool("run_code_cell") == "runCodeCell"

    @pytest.mark.asyncio
    async def test_schema_cache_is_refreshed_for_a_new_connection(self, frontend):
        """A reconnecting tab gets a fresh client and must be re-read."""
        proxy = FakeProxyClient(frontend)
        tools = ProxyTools(proxy)
        await tools.schemas()
        replacement = FakeFrontend()
        proxy.proxy_mcp_client = replacement
        await tools.schemas()
        assert tools._schemas_for is replacement

    @pytest.mark.asyncio
    async def test_raises_when_the_frontend_is_absent(self):
        tools = ProxyTools(FakeProxyClient(None, connected=False))
        with pytest.raises(NotConnectedError):
            await tools.get_cells()

    @pytest.mark.asyncio
    async def test_accepts_a_lazy_proxy_client(self, frontend):
        """The session builds its client after the tools are constructed."""
        holder = {"client": None}
        tools = ProxyTools(lambda: holder["client"])
        assert tools.is_connected() is False
        holder["client"] = FakeProxyClient(frontend)
        assert tools.is_connected() is True


# --------------------------------------------------------------------------
# result / cell normalization
# --------------------------------------------------------------------------


class TestNormalization:
    def test_unwraps_fastmcps_result_envelope(self):
        assert result_payload(FakeToolResult(structured_content={"result": 7})) == 7

    def test_keeps_a_multi_key_payload(self):
        payload = {"cells": [], "extra": 1}
        assert result_payload(FakeToolResult(structured_content=payload)) == payload

    def test_normalizes_the_live_get_cells_shape(self):
        """source arrives as a list of lines and the id lives under `id`."""
        payload = {
            "cells": [
                {
                    "cell_type": "code",
                    "id": "abc",
                    "source": ["print('x')\n", "1 + 1"],
                    "outputs": [],
                }
            ]
        }
        cells = normalize_cells(payload)
        assert cells == [
            {
                "index": 0,
                "id": "abc",
                "source": "print('x')\n1 + 1",
                "outputs": [],
                "raw": payload["cells"][0],
            }
        ]

    def test_accepts_a_bare_list(self):
        assert normalize_cells([{"id": "a", "source": "x"}])[0]["id"] == "a"

    def test_returns_empty_for_an_unrecognised_shape(self):
        assert normalize_cells(None) == []
        assert normalize_cells("nonsense") == []

    def test_reads_the_new_cell_id_reply(self):
        """The frontend replies newCellId; reading only cellId regressed once."""
        assert kernel._cell_id_from_result({"newCellId": "xyz"}) == "xyz"


class TestSplitOutputs:
    def test_separates_stdout_and_stderr(self):
        out = split_outputs([stream_output("a\n"), stream_output("b\n", name="stderr")])
        assert out["stream"] == "a\n"
        assert out["stderr"] == "b\n"

    def test_execute_result_is_not_treated_as_display_data(self):
        """Regression: a cell's return value is current, so lumping it in with
        display_data wrongly flagged trustworthy output as stale."""
        out = split_outputs([execute_result("42")])
        assert out["result"] == "42"
        assert out["display_data"] == ""
        assert out["has_display_data"] is False

    def test_display_data_is_tracked_separately(self):
        out = split_outputs([display_data("  0%|  | 0/40")])
        assert out["display_data"] == "  0%|  | 0/40"
        assert out["has_display_data"] is True

    def test_errors_use_the_traceback(self):
        out = split_outputs([error_output("ValueError", "boom")])
        assert "ValueError: boom" in out["error"]

    def test_handles_mixed_outputs_from_a_real_cell(self):
        out = split_outputs([stream_output("probe\n"), execute_result("42")])
        assert out["stream"] == "probe\n"
        assert out["result"] == "42"

    def test_tolerates_plain_strings(self):
        assert split_outputs(["hello"])["stream"] == "hello"

    def test_empty_outputs(self):
        assert split_outputs([])["stream"] == ""


class TestTail:
    def test_keeps_the_last_lines(self):
        assert tail("\n".join(str(i) for i in range(100)), max_lines=3) == "97\n98\n99"

    def test_caps_characters(self):
        assert tail("x" * 500, max_chars=10) == "x" * 10

    def test_short_text_is_untouched(self):
        assert tail("hi", max_lines=10, max_chars=10) == "hi"

    def test_none_limits_are_ignored(self):
        assert tail("abc", None, None) == "abc"


class TestPayloadMarkers:
    def test_round_trips_a_payload(self):
        text = f"noise\n{kernel.PAYLOAD_BEGIN}{{\"a\": 1}}{kernel.PAYLOAD_END}\n"
        assert extract_payload(text) == {"a": 1}
        assert strip_payload(text) == "noise"

    def test_absent_payload(self):
        assert extract_payload("just output") is None

    def test_malformed_payload_is_not_fatal(self):
        assert extract_payload(f"{kernel.PAYLOAD_BEGIN}not json{kernel.PAYLOAD_END}") is None


class TestStripNotebookOutputs:
    def test_clears_code_cell_outputs_only(self):
        nb = {
            "cells": [
                {"cell_type": "code", "outputs": [1], "execution_count": 3},
                {"cell_type": "markdown", "source": "hi"},
            ]
        }
        stripped = strip_notebook_outputs(nb)
        assert stripped["cells"][0]["outputs"] == []
        assert stripped["cells"][0]["execution_count"] is None
        assert "outputs" not in stripped["cells"][1]


# --------------------------------------------------------------------------
# ProxyTools against the fake frontend
# --------------------------------------------------------------------------


class TestProxyTools:
    @pytest.mark.asyncio
    async def test_add_code_cell_supplies_the_required_args(self, tools, frontend):
        """cellIndex and language are required; omitting them broke the call."""
        cell_id = await tools.add_code_cell("x = 1")
        name, args = frontend.calls[-1]
        assert name == "add_code_cell"
        assert set(args) == {"cellIndex", "language", "code"}
        assert args["language"] == "python"
        assert frontend.find(cell_id) is not None

    @pytest.mark.asyncio
    async def test_add_code_cell_appends_by_default(self, tools, frontend):
        frontend.add("first")
        cell_id = await tools.add_code_cell("second")
        assert frontend.ids()[-1] == cell_id

    @pytest.mark.asyncio
    async def test_add_code_cell_honours_an_explicit_index(self, tools, frontend):
        frontend.add("first")
        cell_id = await tools.add_code_cell("zeroth", index=0)
        assert frontend.ids()[0] == cell_id

    @pytest.mark.asyncio
    async def test_add_code_cell_recovers_an_unreported_id(self, tools, frontend):
        """Fallback path when the reply carries no usable id."""

        original = frontend.call_tool

        async def call_tool(name, arguments=None, timeout=None):
            result = await original(name, arguments, timeout)
            if name == "add_code_cell":
                return FakeToolResult(structured_content={"ok": True})
            return result

        frontend.call_tool = call_tool
        cell_id = await tools.add_code_cell("x = 1")
        assert frontend.find(cell_id) is not None

    @pytest.mark.asyncio
    async def test_fetches_one_cell_by_index_range(self, tools, frontend):
        """Polling must not drag every other cell's source along with it."""
        frontend.add("a")
        target = frontend.add("b", outputs=[stream_output("hi\n")])
        frontend.add("c")
        await tools.get_cells()  # fill the index cache
        frontend.calls.clear()

        cell = await tools.cell_with_outputs(target)

        assert cell["id"] == target
        _, args = frontend.calls[-1]
        assert args["cellIndexStart"] == 1 and args["cellIndexEnd"] == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_a_full_scan_when_the_cell_moved(self, tools, frontend):
        a = frontend.add("a")
        target = frontend.add("b", outputs=[stream_output("hi\n")])
        await tools.get_cells()
        frontend.cells.remove(frontend.find(a))  # target is now at index 0

        cell = await tools.cell_with_outputs(target)
        assert cell["id"] == target

    @pytest.mark.asyncio
    async def test_missing_cell_yields_empty_output(self, tools):
        assert await tools.cell_output("nope") == kernel.EMPTY_OUTPUT

    @pytest.mark.asyncio
    async def test_delete_forgets_the_cached_index(self, tools, frontend):
        cell_id = frontend.add("a")
        await tools.get_cells()
        await tools.delete_cell(cell_id)
        assert str(cell_id) not in tools._index_cache


# --------------------------------------------------------------------------
# asynchronous execution
# --------------------------------------------------------------------------


class TestExecutionManager:
    @pytest.mark.asyncio
    async def test_launch_returns_before_the_cell_finishes(self, tools, frontend):
        """The whole point: a long cell must not have to fit in a request timeout."""
        frontend.run_delay = 0.3
        cell_id = frontend.add("slow")
        manager = ExecutionManager(tools)

        execution = await manager.launch(cell_id)

        assert execution.state == "running"
        assert (await manager.status(execution.exec_id))["state"] == "running"
        await execution.task
        assert (await manager.status(execution.exec_id))["state"] == "done"

    @pytest.mark.asyncio
    async def test_status_is_reachable_by_cell_id(self, tools, frontend):
        cell_id = frontend.add("x")
        manager = ExecutionManager(tools)
        execution = await manager.launch(cell_id)
        await execution.task
        assert (await manager.status(cell_id))["execId"] == execution.exec_id

    @pytest.mark.asyncio
    async def test_unknown_handle(self, tools):
        assert (await ExecutionManager(tools).status("nope"))["state"] == "unknown"

    @pytest.mark.asyncio
    async def test_runs_are_serialized_so_queue_depth_is_meaningful(self, tools, frontend):
        frontend.run_delay = 0.3
        first = frontend.add("a")
        second = frontend.add("b")
        manager = ExecutionManager(tools)

        one = await manager.launch(first)
        two = await manager.launch(second)

        assert one.state == "running"
        assert two.state == "queued"
        assert len(manager.live()) == 2
        await asyncio.gather(one.task, two.task)

    @pytest.mark.asyncio
    async def test_status_includes_the_output_tail(self, tools, frontend):
        frontend.runner = lambda code: [stream_output("\n".join(str(i) for i in range(50)))]
        cell_id = frontend.add("noisy")
        manager = ExecutionManager(tools)
        execution = await manager.launch(cell_id)
        await execution.task

        status = await manager.status(execution.exec_id, tail_lines=3)
        assert status["output_tail"].splitlines() == ["47", "48", "49"]

    @pytest.mark.asyncio
    async def test_a_cell_error_downgrades_a_clean_run(self, tools, frontend):
        frontend.runner = lambda code: [error_output("ValueError", "boom")]
        cell_id = frontend.add("bad")
        manager = ExecutionManager(tools)
        execution = await manager.launch(cell_id)
        await execution.task

        status = await manager.status(execution.exec_id)
        assert status["state"] == "error"
        assert "ValueError" in status["cell_error"]

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_reported(self, tools, frontend):
        cell_id = frontend.add("x")
        frontend.fail_tools = {"run_code_cell"}
        manager = ExecutionManager(tools)
        execution = await manager.launch(cell_id)
        await execution.task
        assert execution.state == "error"
        assert "RuntimeError" in execution.error

    @pytest.mark.asyncio
    async def test_drop_pending_cancels_only_queued_runs(self, tools, frontend):
        frontend.run_delay = 0.3
        one = await (manager := ExecutionManager(tools)).launch(frontend.add("a"))
        two = await manager.launch(frontend.add("b"))

        dropped = manager.drop_pending()

        assert dropped == [two.exec_id]
        assert two.state == "interrupted"
        assert one.state == "running"
        await one.task

    @pytest.mark.asyncio
    async def test_completion_does_not_overwrite_the_interrupted_state(
        self, tools, frontend
    ):
        """Regression: an interrupted cell still returns from run_code_cell (with
        a KeyboardInterrupt in its outputs), which used to clobber the state back
        to done and then get reported as a spurious 'error'."""
        frontend.run_delay = 0.2
        frontend.runner = lambda code: [error_output("KeyboardInterrupt", "")]
        manager = ExecutionManager(tools)
        execution = await manager.launch(frontend.add("busy"))

        manager.mark_interrupted()
        await execution.task

        assert execution.state == "interrupted"
        assert (await manager.status(execution.exec_id))["state"] == "interrupted"


# --------------------------------------------------------------------------
# transient execution and inspection
# --------------------------------------------------------------------------


class TestExecuteTransient:
    @pytest.mark.asyncio
    async def test_leaves_no_cell_behind(self, kernel_tools, frontend):
        frontend.runner = lambda code: [stream_output("hello\n")]
        before = frontend.ids()

        result = await kernel_tools.execute_transient("print('hello')")

        assert result["stdout"] == "hello\n"
        assert frontend.ids() == before

    @pytest.mark.asyncio
    async def test_cleans_up_after_a_failing_cell(self, kernel_tools, frontend):
        frontend.runner = lambda code: [error_output("ValueError", "boom")]
        before = frontend.ids()

        result = await kernel_tools.execute_transient("raise ValueError('boom')")

        assert "ValueError" in result["error"]
        assert frontend.ids() == before

    @pytest.mark.asyncio
    async def test_cleans_up_after_a_timeout(self, kernel_tools, frontend):
        frontend.run_delay = 5
        before = frontend.ids()

        result = await kernel_tools.execute_transient("sleep", timeout_s=0.05)

        assert result["timed_out"] is True
        assert frontend.ids() == before

    @pytest.mark.asyncio
    async def test_separates_stderr_from_stdout(self, kernel_tools, frontend):
        frontend.runner = lambda code: [
            stream_output("out\n"),
            stream_output("err\n", name="stderr"),
        ]
        result = await kernel_tools.execute_transient("x")
        assert result["stdout"] == "out\n"
        assert result["stderr"] == "err\n"

    @pytest.mark.asyncio
    async def test_helper_payload_is_hidden_from_stdout(self, kernel_tools, frontend):
        frontend.runner = lambda code: [
            stream_output(f"user output\n{kernel.PAYLOAD_BEGIN}{{\"a\":1}}{kernel.PAYLOAD_END}")
        ]
        result = await kernel_tools.execute_transient("x")
        assert result["payload"] == {"a": 1}
        assert result["stdout"] == "user output"


class TestEvalExpression:
    @pytest.mark.asyncio
    async def test_returns_the_helper_payload(self, kernel_tools, frontend):
        frontend.runner = lambda code: [
            stream_output(
                kernel.PAYLOAD_BEGIN
                + json.dumps({"repr": "42", "type": "int", "truncated": False})
                + kernel.PAYLOAD_END
            )
        ]
        assert (await kernel_tools.eval_expression("6*7"))["repr"] == "42"

    @pytest.mark.asyncio
    async def test_the_expression_is_embedded_safely(self, kernel_tools, frontend):
        """The expression is JSON-quoted, so quotes in it cannot break the helper."""
        seen = {}

        def runner(code):
            seen["code"] = code
            return [stream_output(kernel.PAYLOAD_BEGIN + "{}" + kernel.PAYLOAD_END)]

        frontend.runner = runner
        await kernel_tools.eval_expression("d['key']")
        assert json.dumps("d['key']") in seen["code"]

    @pytest.mark.asyncio
    async def test_a_missing_payload_is_an_error(self, kernel_tools, frontend):
        frontend.runner = lambda code: [stream_output("nothing useful")]
        with pytest.raises(RuntimeError, match="no payload"):
            await kernel_tools.eval_expression("1")


class TestRuntimeInfo:
    @pytest.mark.asyncio
    async def test_reports_connection_state(self, kernel_tools, frontend):
        frontend.runner = lambda code: [
            stream_output(
                kernel.PAYLOAD_BEGIN
                + json.dumps({"accelerator": "GPU", "gpu_name": "Tesla T4"})
                + kernel.PAYLOAD_END
            )
        ]
        info = await kernel_tools.runtime_info()
        assert info["gpu_name"] == "Tesla T4"
        assert info["connected"] is True
        assert info["runtime_attached"] is True

    @pytest.mark.asyncio
    async def test_does_not_provision_a_runtime_it_just_deleted(
        self, kernel_tools, frontend
    ):
        """runtime_info reads as a status check, so it must not run a cell on
        a notebook with no runtime: that allocates hardware and restarts billing."""
        runs = []
        frontend.runner = lambda code: runs.append(code) or [
            stream_output(kernel.PAYLOAD_BEGIN + '{"accelerator": "GPU"}' + kernel.PAYLOAD_END)
        ]
        await kernel_tools.delete_runtime()

        info = await kernel_tools.runtime_info()

        assert info["runtime_attached"] is False
        assert "no VM was started" in info["note"]
        assert runs == []  # nothing executed, so nothing was provisioned

    @pytest.mark.asyncio
    async def test_the_next_execution_clears_the_deleted_state(
        self, kernel_tools, frontend
    ):
        frontend.runner = _payload_runner(
            [{"ok": 1}, {"accelerator": "GPU", "gpu_name": "A100"}]
        )
        await kernel_tools.delete_runtime()
        # The caller chose to run code, which is what starts a fresh runtime.
        await kernel_tools.execute_transient("1")

        info = await kernel_tools.runtime_info()
        assert info["runtime_attached"] is True
        assert info["gpu_name"] == "A100"

    @pytest.mark.asyncio
    async def test_the_toolbar_says_nothing_is_attached(
        self, kernel_tools, frontend, browser
    ):
        browser.runtime_attached = AsyncMock(return_value=False)
        runs = []
        frontend.runner = lambda code: runs.append(code) or [
            stream_output(kernel.PAYLOAD_BEGIN + '{"accelerator": "GPU"}' + kernel.PAYLOAD_END)
        ]

        info = await kernel_tools.runtime_info()

        assert info["runtime_attached"] is False
        assert runs == []

    @pytest.mark.asyncio
    async def test_the_toolbar_overrides_a_stale_deleted_flag(
        self, kernel_tools, frontend, browser
    ):
        """The user may have reconnected from the UI after our delete."""
        await kernel_tools.delete_runtime()
        browser.runtime_attached = AsyncMock(return_value=True)
        frontend.runner = _payload_runner([{"accelerator": "GPU", "gpu_name": "L4"}])

        info = await kernel_tools.runtime_info()

        assert info["gpu_name"] == "L4"
        assert info["runtime_attached"] is True

    @pytest.mark.asyncio
    async def test_without_a_browser_the_flag_alone_decides(self, frontend):
        tools = KernelTools(FakeProxyClient(frontend), browser=None)
        assert await tools.runtime_attached() is None


class TestGetCellOutput:
    @pytest.mark.asyncio
    async def test_returns_only_the_tail(self, kernel_tools, frontend):
        cell_id = frontend.add(
            "noisy", outputs=[stream_output("\n".join(f"line {i}" for i in range(500)))]
        )
        out = await kernel_tools.get_cell_output(cell_id, tail_lines=3)
        assert out["stream"].splitlines() == ["line 497", "line 498", "line 499"]

    @pytest.mark.asyncio
    async def test_display_data_on_a_running_cell_is_flagged_stale(
        self, kernel_tools, frontend
    ):
        """The documented defect: a progress bar reads 0/398 for ten minutes."""
        frontend.run_delay = 0.5
        cell_id = frontend.add("bar", outputs=[display_data("  0%|  | 0/398")])
        await kernel_tools.executions.launch(cell_id)

        out = await kernel_tools.get_cell_output(cell_id)

        assert out["running"] is True
        assert out["display_data_current"] is False

    @pytest.mark.asyncio
    async def test_display_data_on_an_idle_cell_is_current(self, kernel_tools, frontend):
        cell_id = frontend.add("bar", outputs=[display_data("100%|##| 398/398")])
        out = await kernel_tools.get_cell_output(cell_id)
        assert out["display_data_current"] is True

    @pytest.mark.asyncio
    async def test_a_cell_with_no_rich_output_is_current(self, kernel_tools, frontend):
        cell_id = frontend.add("plain", outputs=[stream_output("hi\n")])
        out = await kernel_tools.get_cell_output(cell_id)
        assert out["display_data_current"] is True

    @pytest.mark.asyncio
    async def test_unknown_cell(self, kernel_tools):
        assert (await kernel_tools.get_cell_output("nope"))["error"] == "cell not found"


class TestNotebookOutline:
    @pytest.mark.asyncio
    async def test_omits_sources_and_outputs(self, kernel_tools, frontend):
        frontend.add("print('a')\nprint('b')", outputs=[stream_output("a\nb\n")])
        outline = await kernel_tools.notebook_outline()
        assert outline[0]["source_head"] == "print('a')"
        assert outline[0]["source_lines"] == 2
        assert outline[0]["output_chars"] == 4
        assert "source" not in outline[0] and "outputs" not in outline[0]

    @pytest.mark.asyncio
    async def test_flags_errors(self, kernel_tools, frontend):
        frontend.add("bad", outputs=[error_output("ValueError", "x")])
        assert (await kernel_tools.notebook_outline())[0]["has_error"] is True

    @pytest.mark.asyncio
    async def test_is_cheaper_than_a_full_listing(self, kernel_tools, frontend):
        for i in range(10):
            frontend.add(f"# cell {i}\n" + "x = 1\n" * 50, outputs=[stream_output("y\n" * 200)])
        outline = json.dumps(await kernel_tools.notebook_outline())
        full = json.dumps(await kernel_tools.tools.get_cells(include_outputs=True))
        assert len(outline) < len(full) / 5


# --------------------------------------------------------------------------
# kernel status and control
# --------------------------------------------------------------------------


class TestKernelStatus:
    @pytest.mark.asyncio
    async def test_idle(self, kernel_tools):
        assert (await kernel_tools.kernel_status()) == {
            "state": "idle",
            "queue_depth": 0,
            "tracked": True,
        }

    @pytest.mark.asyncio
    async def test_busy_names_the_running_cell(self, kernel_tools, frontend):
        frontend.run_delay = 0.4
        cell_id = frontend.add("slow")
        execution = await kernel_tools.executions.launch(cell_id)

        status = await kernel_tools.kernel_status()

        assert status["state"] == "busy"
        assert status["current_cellId"] == cell_id
        assert status["tracked"] is True
        await execution.task

    @pytest.mark.asyncio
    async def test_disconnected(self, browser):
        tools = KernelTools(FakeProxyClient(None, connected=False), browser=browser)
        assert (await tools.kernel_status())["state"] == "disconnected"

    @pytest.mark.asyncio
    async def test_detects_a_cell_started_outside_this_server(self, kernel_tools, frontend):
        """A cell run through the raw tool is not tracked, so say so."""
        frontend.add("x")
        frontend.cells[0]["state"] = "running"

        status = await kernel_tools.kernel_status()

        assert status["state"] == "busy"
        assert status["tracked"] is False


class TestInterrupt:
    @pytest.mark.asyncio
    async def test_reports_that_a_cell_was_running(self, kernel_tools, frontend, browser):
        frontend.run_delay = 0.4
        execution = await kernel_tools.executions.launch(frontend.add("slow"))

        result = await kernel_tools.interrupt()

        assert result == {"interrupted": True, "was_running": True, "cancelled_pending": []}
        browser.interrupt_execution.assert_awaited_once()
        await execution.task
        assert execution.state == "interrupted"

    @pytest.mark.asyncio
    async def test_drops_the_queue_by_default(self, kernel_tools, frontend):
        """Killing a sweep must not let the next queued cell start immediately."""
        frontend.run_delay = 0.4
        one = await kernel_tools.executions.launch(frontend.add("a"))
        two = await kernel_tools.executions.launch(frontend.add("b"))

        result = await kernel_tools.interrupt()

        assert result["cancelled_pending"] == [two.exec_id]
        await one.task

    @pytest.mark.asyncio
    async def test_can_keep_the_queue(self, kernel_tools, frontend):
        frontend.run_delay = 0.4
        one = await kernel_tools.executions.launch(frontend.add("a"))
        two = await kernel_tools.executions.launch(frontend.add("b"))

        result = await kernel_tools.interrupt(cancel_pending=False)

        assert result["cancelled_pending"] == []
        assert two.state == "queued"
        await asyncio.gather(one.task, two.task)

    @pytest.mark.asyncio
    async def test_without_a_browser_it_says_so(self, frontend):
        tools = KernelTools(FakeProxyClient(frontend), browser=None)
        result = await tools.interrupt()
        assert result["interrupted"] is False
        assert "browser" in result["error"]

    @pytest.mark.asyncio
    async def test_reports_a_failed_interrupt(self, kernel_tools, browser):
        browser.interrupt_execution = AsyncMock(return_value=False)
        assert (await kernel_tools.interrupt())["interrupted"] is False


class TestRestartKernel:
    @pytest.mark.asyncio
    async def test_clears_tracked_executions(self, kernel_tools, frontend, browser):
        frontend.run_delay = 0.4
        execution = await kernel_tools.executions.launch(frontend.add("a"))

        assert (await kernel_tools.restart_kernel()) == {"restarted": True}

        browser.restart_kernel.assert_awaited_once()
        assert execution.state == "interrupted"
        await asyncio.gather(execution.task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_without_a_browser(self, frontend):
        tools = KernelTools(FakeProxyClient(frontend), browser=None)
        assert (await tools.restart_kernel())["restarted"] is False


class TestDeleteRuntime:
    @pytest.mark.asyncio
    async def test_drops_pending_work_the_vm_can_never_report(
        self, kernel_tools, frontend, browser
    ):
        frontend.run_delay = 0.4
        execution = await kernel_tools.executions.launch(frontend.add("a"))

        result = await kernel_tools.delete_runtime()

        assert result["deleted"] is True
        browser.delete_runtime.assert_awaited_once()
        # Nothing may stay pending: the machine is gone, so a caller polling
        # get_execution_status would otherwise wait on an id that never resolves.
        assert execution.state == "interrupted"
        await asyncio.gather(execution.task, return_exceptions=True)

    @pytest.mark.asyncio
    async def test_reports_the_menu_failure(self, kernel_tools, browser):
        browser.delete_runtime = AsyncMock(return_value=False)
        result = await kernel_tools.delete_runtime()
        assert result["deleted"] is False
        assert "runtime connected" in result["note"]
        assert "error" not in result

    @pytest.mark.asyncio
    async def test_reports_the_real_cause_when_the_menu_is_blocked(
        self, kernel_tools, browser
    ):
        """An overlay intercepting clicks must not be reported as a missing
        menu entry: that sent a caller debugging connectivity while the VM
        they were trying to stop kept billing."""
        browser.delete_runtime = AsyncMock(
            side_effect=TimeoutError(
                "Locator.click: Timeout 30000ms exceeded.\n"
                "Call log:\n"
                "  - waiting for get_by_text(\"Runtime\", exact=True).first\n"
                "  - <mwc-dialog open=\"\" class=\"user-secrets-prompt\">…</mwc-dialog> "
                "intercepts pointer events\n"
                "  - retrying click action\n"
            )
        )

        result = await kernel_tools.delete_runtime()

        assert result["deleted"] is False
        assert "user-secrets-prompt" in result["error"]
        assert "intercepts pointer events" in result["error"]
        assert "retrying click action" not in result["error"]
        assert "runtime connected" not in result["note"]
        assert "NOT deleted" in result["note"]
        assert "dismiss_colab_dialog" in result["note"]

    @pytest.mark.asyncio
    async def test_a_blocked_restart_is_reported_the_same_way(
        self, kernel_tools, browser
    ):
        browser.restart_kernel = AsyncMock(
            side_effect=kernel_browser.BlockingDialogError("a Colab dialog is blocking")
        )
        result = await kernel_tools.restart_kernel()
        assert result["restarted"] is False
        assert "BlockingDialogError" in result["error"]

    @pytest.mark.asyncio
    async def test_without_a_browser(self, frontend):
        tools = KernelTools(FakeProxyClient(frontend), browser=None)
        assert (await tools.delete_runtime())["deleted"] is False


# --------------------------------------------------------------------------
# batch execution
# --------------------------------------------------------------------------


class TestRunCells:
    @pytest.mark.asyncio
    async def test_runs_every_cell_in_order(self, kernel_tools, frontend):
        ids = [frontend.add(f"c{i}") for i in range(3)]
        results = await kernel_tools.run_cells(ids)
        assert [r["state"] for r in results] == ["done", "done", "done"]
        assert [r["cellId"] for r in results] == ids

    @pytest.mark.asyncio
    async def test_stops_at_the_first_error_and_marks_the_rest_skipped(
        self, kernel_tools, frontend
    ):
        ids = [frontend.add("ok"), frontend.add("bad"), frontend.add("never")]
        frontend.runner = lambda code: (
            [error_output("RuntimeError", "stop")] if "bad" in code else [stream_output("ok\n")]
        )

        results = await kernel_tools.run_cells(ids)

        assert [r["state"] for r in results] == ["done", "error", "skipped"]

    @pytest.mark.asyncio
    async def test_can_continue_past_an_error(self, kernel_tools, frontend):
        ids = [frontend.add("bad"), frontend.add("ok")]
        frontend.runner = lambda code: (
            [error_output("RuntimeError", "stop")] if "bad" in code else [stream_output("ok\n")]
        )

        results = await kernel_tools.run_cells(ids, stop_on_error=False)

        assert [r["state"] for r in results] == ["error", "done"]

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_an_error_not_a_crash(self, kernel_tools, frontend):
        ids = [frontend.add("a")]
        frontend.fail_tools = {"run_code_cell"}
        results = await kernel_tools.run_cells(ids)
        assert results[0]["state"] == "error"


class TestRunAllBelow:
    @pytest.mark.asyncio
    async def test_starts_at_the_named_cell(self, kernel_tools, frontend):
        ids = [frontend.add(f"c{i}") for i in range(4)]
        results = await kernel_tools.run_all_below(ids[2])
        assert [r["cellId"] for r in results] == ids[2:]

    @pytest.mark.asyncio
    async def test_unknown_cell(self, kernel_tools, frontend):
        frontend.add("a")
        results = await kernel_tools.run_all_below("nope")
        assert results[0]["error"] == "cell not found"


# --------------------------------------------------------------------------
# export and file transfer
# --------------------------------------------------------------------------


NOTEBOOK = {
    "cells": [
        {"cell_type": "code", "source": ["print(1)"], "outputs": [{"x": 1}], "execution_count": 1}
    ],
    "nbformat": 4,
}


class TestExportNotebook:
    @pytest.mark.asyncio
    async def test_saves_first_then_downloads(self, kernel_tools, browser, tmp_path):
        dest = tmp_path / "nb.ipynb"

        async def download(path, **kwargs):
            path = str(path)
            with open(path, "w") as fh:
                json.dump(NOTEBOOK, fh)
            return path

        browser.download_notebook = AsyncMock(side_effect=download)

        result = await kernel_tools.export_notebook(str(dest))

        browser.save_notebook.assert_awaited_once()
        assert result["cells"] == 1
        assert result["bytes"] == os.path.getsize(dest)
        assert result["saved_before_export"] is True
        assert json.load(open(dest))["cells"][0]["outputs"] == [{"x": 1}]

    @pytest.mark.asyncio
    async def test_can_strip_outputs(self, kernel_tools, browser, tmp_path):
        dest = tmp_path / "nb.ipynb"

        async def download(path, **kwargs):
            with open(str(path), "w") as fh:
                json.dump(NOTEBOOK, fh)
            return str(path)

        browser.download_notebook = AsyncMock(side_effect=download)

        await kernel_tools.export_notebook(str(dest), include_outputs=False)

        assert json.load(open(dest))["cells"][0]["outputs"] == []

    @pytest.mark.asyncio
    async def test_reports_an_unsaved_notebook(self, kernel_tools, browser, tmp_path):
        """A failed save means the export may predate recent edits."""
        browser.save_notebook = AsyncMock(return_value=False)

        async def download(path, **kwargs):
            with open(str(path), "w") as fh:
                json.dump(NOTEBOOK, fh)
            return str(path)

        browser.download_notebook = AsyncMock(side_effect=download)

        result = await kernel_tools.export_notebook(str(tmp_path / "nb.ipynb"))
        assert result["saved_before_export"] is False

    @pytest.mark.asyncio
    async def test_without_a_browser(self, frontend, tmp_path):
        tools = KernelTools(FakeProxyClient(frontend), browser=None)
        assert "error" in await tools.export_notebook(str(tmp_path / "nb.ipynb"))


def _payload_runner(responses):
    """Frontend runner replying with successive JSON payloads."""
    queue = list(responses)

    def runner(code):
        value = queue.pop(0) if queue else {}
        if callable(value):
            value = value(code)
        return [stream_output(kernel.PAYLOAD_BEGIN + json.dumps(value) + kernel.PAYLOAD_END)]

    return runner


class TestFileTransfer:
    @pytest.mark.asyncio
    async def test_download_reassembles_chunks_in_order(
        self, kernel_tools, frontend, tmp_path, monkeypatch
    ):
        import base64

        monkeypatch.setattr(kernel, "FILE_CHUNK_BYTES", 4)
        payload = b"abcdefghij"  # 10 bytes -> 3 chunks
        chunks = [payload[i : i + 4] for i in range(0, len(payload), 4)]
        frontend.runner = _payload_runner(
            [{"size": len(payload)}]
            + [{"b64": base64.b64encode(c).decode()} for c in chunks]
        )
        dest = tmp_path / "out.bin"

        result = await kernel_tools.download_file("/content/x.bin", str(dest))

        assert dest.read_bytes() == payload
        assert result["bytes"] == len(payload)
        assert result["remote_bytes"] == len(payload)

    @pytest.mark.asyncio
    async def test_download_reports_a_missing_remote_file(self, kernel_tools, frontend, tmp_path):
        frontend.runner = _payload_runner([{"error": "no such file: /x", "size": 0}])
        result = await kernel_tools.download_file("/x", str(tmp_path / "out.bin"))
        assert "no such file" in result["error"]

    @pytest.mark.asyncio
    async def test_download_refuses_an_oversized_file(self, kernel_tools, frontend, tmp_path):
        frontend.runner = _payload_runner([{"size": kernel.MAX_TRANSFER_BYTES + 1}])
        result = await kernel_tools.download_file("/big", str(tmp_path / "out.bin"))
        assert "limit is" in result["error"]
        assert "COLAB_MCP_MAX_TRANSFER_BYTES" in result["error"]

    def test_the_default_ceiling_fits_an_activation_dump(self):
        """256 MB refused a single .npy of features; the transport was never
        the limit, only this guard was."""
        assert kernel.MAX_TRANSFER_BYTES >= 8 * 1024**3

    @pytest.mark.asyncio
    async def test_a_file_over_the_old_256mb_cap_is_accepted(
        self, kernel_tools, frontend, tmp_path, monkeypatch
    ):
        import base64

        # A "412 MB" file, at a chunk size that keeps the test to three cells.
        size = 412_876_928
        monkeypatch.setattr(kernel, "FILE_CHUNK_BYTES", size // 2 + 1)
        chunks = [b"a" * (size // 2 + 1), b"b" * (size - (size // 2 + 1))]
        frontend.runner = _payload_runner(
            [{"size": size}] + [{"b64": base64.b64encode(c).decode()} for c in chunks]
        )
        dest = tmp_path / "X_005.npy"

        result = await kernel_tools.download_file("/content/features/X_005.npy", str(dest))

        assert "error" not in result
        assert result["bytes"] == size
        assert dest.stat().st_size == size

    @pytest.mark.asyncio
    async def test_upload_sends_every_chunk(
        self, kernel_tools, frontend, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(kernel, "FILE_CHUNK_BYTES", 4)
        src = tmp_path / "in.bin"
        src.write_bytes(b"abcdefghij")
        frontend.runner = _payload_runner([{"written": 4}] * 5)

        result = await kernel_tools.upload_file(str(src), "/content/x.bin")

        assert result["bytes"] == 10

    @pytest.mark.asyncio
    async def test_upload_truncates_before_appending(
        self, kernel_tools, frontend, tmp_path, monkeypatch
    ):
        """The first chunk must open 'wb' so a re-upload does not append to a
        stale file; every later chunk must append."""
        monkeypatch.setattr(kernel, "FILE_CHUNK_BYTES", 4)
        src = tmp_path / "in.bin"
        src.write_bytes(b"abcdefghij")
        modes = []

        def runner(code):
            modes.append('"wb"' if '"wb"' in code else '"ab"')
            return [stream_output(kernel.PAYLOAD_BEGIN + "{}" + kernel.PAYLOAD_END)]

        frontend.runner = runner

        await kernel_tools.upload_file(str(src), "/content/x.bin")

        assert modes[0] == '"wb"'
        assert set(modes[1:]) == {'"ab"'}

    @pytest.mark.asyncio
    async def test_upload_handles_an_empty_file(
        self, kernel_tools, frontend, tmp_path
    ):
        src = tmp_path / "empty.bin"
        src.write_bytes(b"")
        frontend.runner = _payload_runner([{"written": 0}])

        result = await kernel_tools.upload_file(str(src), "/content/empty.bin")

        assert result["bytes"] == 0
