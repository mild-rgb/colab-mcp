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

"""A stand-in for the notebook tools the Colab frontend proxies to us.

The schemas and reply shapes here are the ones a live Colab session was
observed to use, not invented ones:

* `add_code_cell(cellIndex, language, code)` — all three required — replying
  `{"newCellId": ...}`, NOT `{"cellId": ...}`.
* `get_cells(cellIndexStart, cellIndexEnd, includeOutputs)` with `source` as a
  list of lines and the cell id under `id`.
* `run_code_cell(cellId)` replying with the cell's `outputs`.
* Jupyter output records: `stream` (with `name`), `execute_result`,
  `display_data`, `error`.

Tests that drive this fake therefore fail if the adapter layer regresses to
assuming different names or shapes.
"""

from dataclasses import dataclass, field
from typing import Any


FRONTEND_SCHEMAS = {
    "add_code_cell": {
        "type": "object",
        "properties": {
            "cellIndex": {"type": "number"},
            "language": {"type": "string"},
            "code": {"type": "string"},
        },
        "required": ["cellIndex", "language", "code"],
    },
    "add_text_cell": {
        "type": "object",
        "properties": {"cellIndex": {"type": "number"}, "content": {"type": "string"}},
        "required": ["cellIndex", "content"],
    },
    "delete_cell": {
        "type": "object",
        "properties": {"cellId": {"type": "string"}},
        "required": ["cellId"],
    },
    "get_cells": {
        "type": "object",
        "properties": {
            "cellIndexStart": {"type": "number"},
            "cellIndexEnd": {"type": "number"},
            "includeOutputs": {"type": "boolean"},
        },
        "required": [],
    },
    "move_cell": {
        "type": "object",
        "properties": {"cellId": {"type": "string"}, "cellIndex": {"type": "number"}},
        "required": ["cellId", "cellIndex"],
    },
    "run_code_cell": {
        "type": "object",
        "properties": {"cellId": {"type": "string"}},
        "required": ["cellId"],
    },
    "update_cell": {
        "type": "object",
        "properties": {"cellId": {"type": "string"}, "content": {"type": "string"}},
        "required": ["cellId", "content"],
    },
}


@dataclass
class FakeToolResult:
    """Mirrors fastmcp's CallToolResult closely enough for result_payload."""

    structured_content: dict | None = None
    content: list = field(default_factory=list)
    data: Any = None
    is_error: bool = False
    meta: dict | None = None


@dataclass
class FakeToolDef:
    name: str
    inputSchema: dict  # noqa: N815 - matches mcp.types.Tool


def stream_output(text: str, name: str = "stdout") -> dict:
    return {"output_type": "stream", "name": name, "text": [text]}


def execute_result(text: str, count: int = 1) -> dict:
    return {
        "output_type": "execute_result",
        "data": {"text/plain": [text]},
        "execution_count": count,
    }


def display_data(text: str) -> dict:
    return {"output_type": "display_data", "data": {"text/plain": [text]}}


def error_output(ename: str, evalue: str) -> dict:
    return {
        "output_type": "error",
        "ename": ename,
        "evalue": evalue,
        "traceback": [f"{ename}: {evalue}"],
    }


class FakeFrontend:
    """An in-memory notebook exposing the frontend's tool surface."""

    def __init__(self, cells=None):
        self.cells: list[dict] = list(cells or [])
        self.calls: list[tuple[str, dict]] = []
        self._counter = 0
        # name -> callable(code) -> list[output]; the default just echoes.
        self.runner = None
        self.run_delay = 0.0
        self.fail_tools: set[str] = set()

    # -- helpers -----------------------------------------------------------

    def _new_id(self) -> str:
        self._counter += 1
        return f"cell{self._counter}"

    def add(self, code: str = "", outputs=None, cell_id: str | None = None) -> str:
        cid = cell_id or self._new_id()
        self.cells.append(
            {
                "cell_type": "code",
                "id": cid,
                "source": code.splitlines(keepends=True),
                "execution_count": None,
                "outputs": outputs or [],
            }
        )
        return cid

    def find(self, cell_id: str) -> dict | None:
        return next((c for c in self.cells if c["id"] == cell_id), None)

    def ids(self) -> list[str]:
        return [c["id"] for c in self.cells]

    # -- MCP client surface ------------------------------------------------

    async def list_tools(self):
        return [FakeToolDef(name=n, inputSchema=s) for n, s in FRONTEND_SCHEMAS.items()]

    async def call_tool(self, name, arguments=None, timeout=None):
        import asyncio

        args = arguments or {}
        self.calls.append((name, dict(args)))
        if name in self.fail_tools:
            raise RuntimeError(f"frontend rejected {name}")

        if name == "get_cells":
            start = args.get("cellIndexStart")
            end = args.get("cellIndexEnd")
            cells = self.cells
            if start is not None:
                stop = (end if end is not None else start) + 1
                cells = cells[start:stop]
            out = []
            for cell in cells:
                copy = {k: v for k, v in cell.items() if k != "outputs"}
                if args.get("includeOutputs"):
                    copy["outputs"] = cell["outputs"]
                out.append(copy)
            return FakeToolResult(structured_content={"cells": out})

        if name == "add_code_cell":
            for required in ("cellIndex", "language", "code"):
                if required not in args:
                    raise RuntimeError(f"add_code_cell missing required arg {required}")
            cid = self._new_id()
            self.cells.insert(
                min(args["cellIndex"], len(self.cells)),
                {
                    "cell_type": "code",
                    "id": cid,
                    "source": str(args["code"]).splitlines(keepends=True),
                    "execution_count": None,
                    "outputs": [],
                },
            )
            # The real frontend replies with newCellId, not cellId.
            return FakeToolResult(structured_content={"newCellId": cid})

        if name == "run_code_cell":
            cell = self.find(args["cellId"])
            if cell is None:
                raise RuntimeError(f"no such cell {args['cellId']}")
            if self.run_delay:
                await asyncio.sleep(self.run_delay)
            code = "".join(cell["source"])
            cell["outputs"] = self.runner(code) if self.runner else [stream_output("ok\n")]
            cell["execution_count"] = 1
            return FakeToolResult(structured_content={"outputs": cell["outputs"]})

        if name == "delete_cell":
            self.cells = [c for c in self.cells if c["id"] != args["cellId"]]
            return FakeToolResult(structured_content={"result": True})

        if name == "update_cell":
            cell = self.find(args["cellId"])
            if cell is None:
                raise RuntimeError(f"no such cell {args['cellId']}")
            cell["source"] = str(args["content"]).splitlines(keepends=True)
            return FakeToolResult(structured_content={"result": True})

        raise RuntimeError(f"unknown tool {name}")


class FakeProxyClient:
    """Matches the bit of ColabProxyClient that ProxyTools depends on."""

    def __init__(self, frontend: FakeFrontend | None = None, connected: bool = True):
        self.proxy_mcp_client = frontend
        self._connected = connected

    def is_connected(self) -> bool:
        return self._connected and self.proxy_mcp_client is not None

    def disconnect(self):
        self._connected = False
