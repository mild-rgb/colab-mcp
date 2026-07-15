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

"""Playwright-driven Colab browser component.

Launches a Chromium instance the agent can control (open the tokened Colab URL,
click "Connect to Colab", change the runtime accelerator to a GPU, screenshot),
while reusing the user's existing Google login.

Login reuse: the user's snap-chromium profile encrypts cookies with the
`basic` password store (a fixed key), so we copy the login-essential files into
a dedicated data dir and launch snap chromium (same browser version, so no
profile-downgrade issues) with remote debugging on a *non-default* data dir
(Chrome >=136 only permits remote debugging off the default profile). Playwright
then attaches over CDP.
"""

import asyncio
import contextlib
import logging
import os
import shutil
import socket
import subprocess

HOME = os.path.expanduser("~")
CHROMIUM = os.environ.get("COLAB_MCP_CHROMIUM", "/snap/bin/chromium")
# Source profile that already holds the user's Google login.
SRC_PROFILE = os.environ.get(
    "COLAB_MCP_SRC_PROFILE", f"{HOME}/snap/chromium/common/chromium"
)
# Dedicated (non-default) data dir we launch under, kept beside the source so it
# stays inside snap's writable area.
PW_PROFILE = os.environ.get(
    "COLAB_MCP_PW_PROFILE", f"{HOME}/snap/chromium/common/colab-mcp-pw-profile"
)

# Login-essential files to carry over from the source profile.
_TOP_FILES = ["Local State", "First Run"]
_DEFAULT_FILES = [
    "Cookies",
    "Login Data",
    "Login Data For Account",
    "Preferences",
    "Secure Preferences",
    "Web Data",
]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _sync_login_profile() -> None:
    """Copy the login-essential files from the user's profile into PW_PROFILE."""
    os.makedirs(os.path.join(PW_PROFILE, "Default"), exist_ok=True)
    for rel in _TOP_FILES:
        src = os.path.join(SRC_PROFILE, rel)
        if os.path.exists(src):
            with contextlib.suppress(OSError):
                shutil.copy2(src, os.path.join(PW_PROFILE, rel))
    for f in _DEFAULT_FILES:
        src = os.path.join(SRC_PROFILE, "Default", f)
        if os.path.exists(src):
            with contextlib.suppress(OSError):
                shutil.copy2(src, os.path.join(PW_PROFILE, "Default", f))
    src_net = os.path.join(SRC_PROFILE, "Default", "Network")
    if os.path.isdir(src_net):
        with contextlib.suppress(OSError):
            shutil.copytree(
                src_net,
                os.path.join(PW_PROFILE, "Default", "Network"),
                dirs_exist_ok=True,
            )


class ColabBrowser:
    """A Playwright-controlled Chromium bound to a Colab session."""

    def __init__(self):
        self._proc: subprocess.Popen | None = None
        self._pw = None
        self._browser = None
        self.page = None
        self.port: int | None = None
        self._lock = asyncio.Lock()

    def is_open(self) -> bool:
        return self.page is not None and self._proc is not None and self._proc.poll() is None

    async def launch(self, url: str):
        """Launch Chromium on the login-reusing profile and attach over CDP."""
        async with self._lock:
            if self.is_open():
                # already open: just navigate the existing page
                with contextlib.suppress(Exception):
                    await self.page.goto(url, wait_until="domcontentloaded")
                return self.page

            # profile copy and process spawn both do blocking filesystem work
            await asyncio.to_thread(_sync_login_profile)
            self.port = _free_port()
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(PW_PROFILE, "DevToolsActivePort"))

            self._proc = await asyncio.to_thread(
                subprocess.Popen,
                [
                    CHROMIUM,
                    f"--user-data-dir={PW_PROFILE}",
                    f"--remote-debugging-port={self.port}",
                    "--remote-allow-origins=*",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--password-store=basic",
                    url,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logging.info("colab-mcp: launched chromium pid=%s cdp=%s", self._proc.pid, self.port)

            from playwright.async_api import async_playwright

            self._pw = await async_playwright().start()
            endpoint = f"http://127.0.0.1:{self.port}"
            last_err = None
            for _ in range(60):  # up to ~30s for CDP to come up
                if self._proc.poll() is not None:
                    raise RuntimeError("chromium exited before CDP was ready")
                try:
                    self._browser = await self._pw.chromium.connect_over_cdp(endpoint)
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    await asyncio.sleep(0.5)
            if self._browser is None:
                raise RuntimeError(f"could not connect to chromium CDP: {last_err}")

            ctx = self._browser.contexts[0]
            self.page = ctx.pages[0] if ctx.pages else await ctx.new_page()
            return self.page

    async def click_connect(self) -> bool:
        """Click the "Connect to Colab" authorization affordance if present.

        Matches the "Connect to ..." authorization text specifically so we don't
        click the unrelated runtime "Connect" button in the top-right toolbar.
        """
        if self.page is None:
            return False
        with contextlib.suppress(Exception):
            # Only act when the authorization dialog ("Connect to a local Colab
            # MCP server") is actually present, so we never click the unrelated
            # runtime "Connect" button in the top-right toolbar.
            if not await self.page.get_by_text(
                "local Colab MCP server", exact=False
            ).count():
                return False
            # Its primary action is a button whose exact accessible name is
            # "Connect" (distinct from the "Connect separate tabs" expander).
            btn = self.page.get_by_role("button", name="Connect", exact=True)
            if await btn.count() and await btn.first.is_visible():
                await btn.first.click()
                logging.info("colab-mcp: clicked Connect on MCP authorization dialog")
                return True
        return False

    async def set_runtime_accelerator(self, accelerator: str = "T4 GPU") -> str:
        """Automate Runtime -> Change runtime type -> <accelerator> -> Save."""
        if self.page is None:
            return "browser not open"
        p = self.page
        # Locator clicks auto-wait for the target to be visible and actionable,
        # so no fixed sleeps are needed between menu steps.
        await p.get_by_text("Runtime", exact=True).first.click()
        await p.get_by_text("Change runtime type", exact=False).first.click()
        radio = p.get_by_text(accelerator, exact=True).first
        await radio.wait_for(state="visible", timeout=10_000)
        await radio.click()
        # A "Disconnect and delete runtime" confirmation appears only when a
        # runtime is currently connected.
        with contextlib.suppress(Exception):
            ok = p.get_by_text("OK", exact=True).last
            await ok.wait_for(state="visible", timeout=2_000)
            await ok.click()
        save = p.get_by_text("Save", exact=True).last
        await save.click()
        # Wait for the dialog to dismiss so callers observe the applied change.
        with contextlib.suppress(Exception):
            await save.wait_for(state="hidden", timeout=10_000)
        return f"requested runtime accelerator: {accelerator}"

    async def screenshot(self, path: str) -> str:
        if self.page is None:
            return "browser not open"
        await self.page.screenshot(path=path)
        return path

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            if self._browser:
                await self._browser.close()
        with contextlib.suppress(Exception):
            if self._pw:
                await self._pw.stop()
        with contextlib.suppress(Exception):
            if self._proc and self._proc.poll() is None:
                self._proc.terminate()
        self._browser = self._pw = self.page = self._proc = None
