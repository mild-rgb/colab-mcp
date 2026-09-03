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
import json
import logging
import os
import re
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


# Copying the source profile over PW_PROFILE on every launch is what carries a
# fresh Chrome login through to automation. It also destroys any session
# established *inside* PW_PROFILE -- the only kind that can survive unattended
# when the day-to-day profile is signed out. Set COLAB_MCP_SKIP_PROFILE_SYNC=1
# to keep the automation profile self-contained: sign in once in the automation
# window and it persists across launches.
SKIP_PROFILE_SYNC = os.environ.get(
    "COLAB_MCP_SKIP_PROFILE_SYNC", ""
).strip().lower() in ("1", "true", "yes")


# Which Google account to pick when the chooser offers more than one. Read from
# the Chrome profile when unset. colab-mcp never reads or types a password:
# the sign-in flow is clicked through and Chrome's own password manager fills
# the field, exactly as it would for a human driving the same window.
GOOGLE_EMAIL = os.environ.get("COLAB_MCP_GOOGLE_EMAIL", "")

_SIGNIN_HOSTS = ("accounts.google.com", "accounts.youtube.com")

# Colab's menus and Google's sign-in steps need a beat to settle between clicks;
# these waits are unavoidable because the next element does not exist yet to be
# waited on. Scale them up on a slow machine, or down in tests.
UI_PAUSE_SCALE = float(os.environ.get("COLAB_MCP_UI_PAUSE_SCALE", "1"))


async def _pause(seconds: float) -> None:
    await asyncio.sleep(seconds * UI_PAUSE_SCALE)


def drive_id_from_url(url: str) -> str:
    """The Drive file id in a Colab `/drive/<id>` URL, or "" if there is none."""
    match = re.search(r"/drive/([A-Za-z0-9_-]+)", url or "")
    return match.group(1) if match else ""


# Chrome must write downloads somewhere both it and we can see; snap confines it
# to its own area, so a host /tmp path would silently land nowhere reachable.
DOWNLOAD_DIR = os.environ.get(
    "COLAB_MCP_DOWNLOAD_DIR", os.path.join(os.path.dirname(PW_PROFILE), "colab-mcp-downloads")
)


def _profile_email() -> str:
    """The signed-in account recorded in the source Chrome profile, if any."""
    if GOOGLE_EMAIL:
        return GOOGLE_EMAIL
    path = os.path.join(SRC_PROFILE, "Default", "Preferences")
    with contextlib.suppress(Exception):
        with open(path, encoding="utf-8") as handle:
            prefs = json.load(handle)
        for entry in prefs.get("account_info") or []:
            if entry.get("email"):
                return entry["email"]
    return ""


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _sync_login_profile() -> None:
    """Copy the login-essential files from the user's profile into PW_PROFILE.

    No-op when COLAB_MCP_SKIP_PROFILE_SYNC is set, so an automation profile that
    is already signed in is not clobbered by a signed-out source profile.
    """
    if SKIP_PROFILE_SYNC:
        logging.info(
            "colab-mcp: COLAB_MCP_SKIP_PROFILE_SYNC set; keeping the existing login in %s",
            PW_PROFILE,
        )
        return
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
                    # The usual cause is another chromium already holding this
                    # data dir: the second instance hands the URL over and
                    # exits, so CDP never comes up on our port.
                    raise RuntimeError(
                        "chromium exited before CDP was ready; another chromium "
                        f"may already be using the profile at {PW_PROFILE}"
                    )
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

    async def navigate(self, url: str):
        """Point the browser at `url`, launching it first if it isn't open.

        Unlike the navigate-if-already-open path inside `launch`, failures here
        are raised rather than suppressed: callers that navigate to a specific
        notebook need to know when the load did not happen.
        """
        if not self.is_open():
            return await self.launch(url)
        async with self._lock:
            await self.page.goto(url, wait_until="domcontentloaded")
            return self.page

    async def title(self) -> str:
        """Best-effort page title; empty string when unavailable."""
        if self.page is None:
            return ""
        with contextlib.suppress(Exception):
            return await self.page.title()
        return ""

    async def _on_signin_page(self) -> bool:
        url = (self.page.url or "") if self.page else ""
        return any(host in url for host in _SIGNIN_HOSTS)

    async def _needs_signin_on_colab(self) -> bool:
        """We are on a Colab page that is not signed in.

        Two shapes, and only the first used to be checked: Colab's "You must be
        logged in..." interstitial, and -- on a notebook Colab renders
        anonymously, such as the scratch notebook -- no interstitial at all,
        just a "Sign in" button in the top-right toolbar. Missing the second
        meant ensure_signed_in() returned True having done nothing, and the
        session reported as connected while being unusable.
        """
        if self.page is None:
            return False
        with contextlib.suppress(Exception):
            if await self.page.get_by_text(
                "must be logged in with a Google Account", exact=False
            ).count():
                return True
        # Only trust the toolbar affordance on an actual Colab page, so stray
        # "Sign in" text elsewhere cannot strand us in the sign-in loop.
        with contextlib.suppress(Exception):
            if "colab" not in (self.page.url or "").lower():
                return False
            for role in ("button", "link"):
                if await self.page.get_by_role(role, name="Sign in", exact=True).count():
                    return True
        return False

    async def is_signed_in(self) -> bool:
        """Whether the current page is a usable, signed-in Colab session."""
        if self.page is None:
            return False
        return not await self._on_signin_page() and not await self._needs_signin_on_colab()

    async def ensure_signed_in(self, timeout_s: float = 90.0) -> bool:
        """Click through Google's sign-in flow if Colab redirected to it.

        The launch profile carries the user's cookies *and* Chrome's saved
        credentials, so the only thing missing is the clicking: choose the
        account, advance past the email step, and let Chrome's password manager
        fill the password field. We never read or type the password ourselves —
        if Chrome does not autofill, we stop and say so rather than pretending
        to handle it.

        Returns True once the browser is off the sign-in pages.
        """
        if self.page is None:
            return False

        p = self.page
        # Colab may render its own "you must be logged in" interstitial rather
        # than redirecting; clicking its Sign in link starts the normal flow.
        if await self._needs_signin_on_colab():
            logging.info("colab-mcp: Colab is asking us to sign in")
            with contextlib.suppress(Exception):
                link = p.get_by_text("Sign in", exact=True).first
                if await link.count():
                    await link.click()
                    await _pause(4)

        if not await self._on_signin_page() and not await self._needs_signin_on_colab():
            return True

        email = _profile_email()
        logging.info("colab-mcp: Google sign-in required; clicking through")
        deadline = asyncio.get_running_loop().time() + timeout_s

        while asyncio.get_running_loop().time() < deadline:
            if not await self._on_signin_page():
                if await self._needs_signin_on_colab():
                    # Back on Colab but still unauthenticated: click through again.
                    with contextlib.suppress(Exception):
                        link = p.get_by_text("Sign in", exact=True).first
                        if await link.count():
                            await link.click()
                            await _pause(4)
                            continue
                    await _pause(2)
                    continue
                logging.info("colab-mcp: sign-in complete")
                return True

            # "Choose an account" — pick the profile's own account when we can
            # identify it, otherwise the first offered.
            with contextlib.suppress(Exception):
                tile = (
                    p.get_by_text(email, exact=False).first
                    if email
                    else p.locator("[data-identifier], li[class*=account]").first
                )
                if await tile.count() and await tile.first.is_visible():
                    await tile.first.click()
                    await _pause(2)
                    continue

            # Email step: the address is not a secret and Chrome often prefills
            # it anyway; fill it only when the field is empty.
            with contextlib.suppress(Exception):
                field = p.locator("input[type=email]").first
                if await field.count() and await field.is_visible():
                    if email and not (await field.input_value()):
                        await field.fill(email)
                    await p.get_by_role("button", name="Next", exact=False).first.click()
                    await _pause(3)
                    continue

            # Password step: clicking the field is a trusted gesture over CDP,
            # which is what prompts Chrome to autofill the saved credential.
            with contextlib.suppress(Exception):
                field = p.locator("input[type=password]").first
                if await field.count() and await field.is_visible():
                    await field.click()
                    await _pause(2)
                    if await field.input_value():
                        await field.press("Enter")
                        await _pause(4)
                        continue
                    logging.warning(
                        "colab-mcp: Chrome did not autofill the saved password; "
                        "the user needs to complete sign-in in the browser window"
                    )
                    return False

            await _pause(2)

        logging.warning("colab-mcp: still on the sign-in flow after %ss", timeout_s)
        return not await self._on_signin_page()

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

    async def _runtime_menu(self, item: str, timeout_ms: int = 10_000) -> bool:
        """Open the Runtime menu and click the entry whose text contains `item`."""
        if self.page is None:
            return False
        p = self.page
        await p.get_by_text("Runtime", exact=True).first.click()
        entry = p.get_by_text(item, exact=False).first
        await entry.wait_for(state="visible", timeout=timeout_ms)
        await entry.click()
        return True

    async def _confirm_dialog(self, *labels: str, timeout_ms: int = 3_000) -> bool:
        """Click the first of `labels` that shows up in a confirmation dialog."""
        if self.page is None:
            return False
        for label in labels:
            with contextlib.suppress(Exception):
                button = self.page.get_by_text(label, exact=True).last
                await button.wait_for(state="visible", timeout=timeout_ms)
                await button.click()
                return True
        return False

    async def interrupt_execution(self) -> bool:
        """Runtime > Interrupt execution — a kernel SIGINT, not a restart.

        Deliberately distinct from `set_runtime_accelerator`: interrupting must
        leave the kernel's globals (a loaded multi-gigabyte model, say) intact,
        which a runtime restart would discard.
        """
        if self.page is None:
            return False
        try:
            await self._runtime_menu("Interrupt execution")
            return True
        except Exception:  # noqa: BLE001
            logging.exception("colab-mcp: could not interrupt execution via the menu")
            # The keyboard shortcut works even when the menu item is disabled or
            # the menu failed to open.
            with contextlib.suppress(Exception):
                await self.page.keyboard.press("Control+m")
                await self.page.keyboard.press("i")
                return True
            return False

    async def restart_kernel(self) -> bool:
        """Runtime > Restart session, confirming the prompt if one appears."""
        if self.page is None:
            return False
        try:
            # Colab has renamed this over time; match either wording, and avoid
            # "Restart and run all" / "Disconnect and delete runtime".
            for label in ("Restart session", "Restart runtime"):
                with contextlib.suppress(Exception):
                    await self._runtime_menu(label, timeout_ms=3_000)
                    break
            else:
                return False
            await self._confirm_dialog("Yes", "OK", "Restart")
            return True
        except Exception:  # noqa: BLE001
            logging.exception("colab-mcp: could not restart the kernel")
            return False

    async def _file_menu(self, item: str, timeout_ms: int = 10_000):
        """Open the File menu and return the menu item whose name contains `item`.

        Colab's menus are `role=menuitem` entries whose text carries a shortcut
        or a submenu arrow ("Download ►", "Save Ctrl+S"), so they are matched by
        accessible name substring rather than exact text.
        """
        p = self.page
        await p.get_by_text("File", exact=True).first.click()
        entry = p.get_by_role("menuitem", name=item).first
        await entry.wait_for(state="visible", timeout=timeout_ms)
        return entry

    async def save_notebook(self) -> bool:
        """Force a save so an export does not pick up a stale autosave."""
        if self.page is None:
            return False
        # File > Save, rather than Ctrl+S, which silently does nothing when the
        # editor does not have keyboard focus.
        try:
            entry = await self._file_menu("Save", timeout_ms=8_000)
            await entry.click()
        except Exception:  # noqa: BLE001
            logging.exception("colab-mcp: could not save via the File menu")
            with contextlib.suppress(Exception):
                await self.page.keyboard.press("Escape")
            return False
        # The save is a background request; give it a moment to land before a
        # download reads the stored copy.
        await _pause(3)
        return True

    async def upload_notebook(self, src_path: str, timeout_s: float = 180.0) -> str:
        """File > Upload notebook, returning the Drive id Colab lands on.

        The inverse of `download_notebook`: a local .ipynb becomes a real
        notebook in Drive rather than bytes on the VM's scratch disk. Colab
        navigates to the uploaded notebook, so the caller must re-attach the
        bridge afterwards — see ColabSessionProxy.import_notebook.
        """
        if self.page is None:
            raise RuntimeError("browser not open")
        if not os.path.isfile(src_path):
            raise RuntimeError(f"no such file: {src_path}")
        p = self.page

        # Uploading always creates a *new* Drive file, so the notebook we are
        # already on is the one thing the landing URL cannot be. Without this,
        # the poll below matches the current URL instantly and reports the old
        # notebook's id.
        previous_id = drive_id_from_url(p.url)

        entry = await self._file_menu("Upload notebook")
        await entry.click()
        await _pause(2)

        # The dialog's file input is usually present but hidden, and
        # set_input_files drives it without involving the OS file chooser.
        # Where it is absent, clicking Browse raises a real chooser instead.
        chosen = False
        with contextlib.suppress(Exception):
            file_input = p.locator("input[type=file]").first
            if await file_input.count():
                await file_input.set_input_files(src_path)
                chosen = True
        if not chosen:
            async with p.expect_file_chooser(timeout=15_000) as chooser_info:
                await p.get_by_text("Browse", exact=False).first.click()
            chooser = await chooser_info.value
            await chooser.set_files(src_path)

        # Colab uploads to Drive and navigates to the new notebook.
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        while loop.time() < deadline:
            await asyncio.sleep(1)
            notebook_id = drive_id_from_url(p.url)
            if notebook_id and notebook_id != previous_id:
                logging.info("colab-mcp: uploaded notebook is %s", notebook_id)
                return notebook_id
        raise RuntimeError(
            f"Colab did not open an uploaded notebook within {timeout_s}s "
            f"(still at {p.url!r})"
        )

    async def _settle_download(self, directory: str, before: set[str], timeout_s: float):
        """Wait for a new, finished file to appear in `directory`."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        candidate = None
        while loop.time() < deadline:
            await asyncio.sleep(0.5)
            names = set(os.listdir(directory)) - before
            # .crdownload is Chrome's in-progress marker.
            done = [n for n in names if not n.endswith(".crdownload")]
            if done:
                candidate = os.path.join(directory, sorted(done)[0])
                size = os.path.getsize(candidate)
                await asyncio.sleep(0.5)
                if size and os.path.getsize(candidate) == size:
                    return candidate
        if candidate:
            return candidate
        raise RuntimeError(f"no download appeared in {directory} within {timeout_s}s")

    async def download_notebook(self, dest_path: str, timeout_ms: int = 120_000) -> str:
        """File > Download > Download .ipynb, saved straight to `dest_path`.

        The bytes go browser -> disk without passing through the agent, which is
        the entire point: a 250 KB executed notebook costs nothing in context.

        Playwright's own download API is unusable here: snap-confined chromium
        writes into its private mount namespace, so the artifact path Playwright
        reports does not exist on the host. Instead we point Chrome's download
        directory at somewhere inside snap's writable area — which *is* visible
        to both — and collect the file from there.
        """
        if self.page is None:
            raise RuntimeError("browser not open")
        p = self.page
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)) or ".", exist_ok=True)
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        before = set(os.listdir(DOWNLOAD_DIR))

        cdp = await p.context.new_cdp_session(p)
        await cdp.send(
            "Browser.setDownloadBehavior",
            {"behavior": "allow", "downloadPath": DOWNLOAD_DIR, "eventsEnabled": True},
        )

        # "Download" is a submenu; hovering expands it. Its own entry sorts
        # before the ".ipynb"/".py" children, so .first is the parent.
        parent = await self._file_menu("Download")
        await parent.hover()
        ipynb = p.get_by_role("menuitem", name="Download .ipynb").first
        await ipynb.wait_for(state="visible", timeout=10_000)
        await ipynb.click()

        src = await self._settle_download(DOWNLOAD_DIR, before, timeout_ms / 1000)
        shutil.move(src, dest_path)
        size = os.path.getsize(dest_path)
        if size == 0:
            raise RuntimeError(f"downloaded notebook {src!r} was empty")
        logging.info("colab-mcp: downloaded %s (%s bytes)", dest_path, size)
        return dest_path

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
