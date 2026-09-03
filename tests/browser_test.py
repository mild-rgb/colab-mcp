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

"""Regression tests for the Playwright-driven Colab browser.

The menu strings and selectors here are the ones a live Colab session uses.
Colab renders menu entries as `role=menuitem` whose text carries a shortcut or
a submenu arrow ("Download ►", "Save Ctrl+S"), so exact-text matching does not
work — that is pinned below because it broke the export path once already.
"""

import asyncio
import json
import os
import time

import pytest

from colab_mcp import browser as browser_mod
from colab_mcp.browser import ColabBrowser


class FakeLocator:
    def __init__(self, page, key, *, count=1, visible=True, value="", text=""):
        self.page = page
        self.key = key
        self._count = count
        self._visible = visible
        self._value = value
        self._text = text

    @property
    def first(self):
        return self

    @property
    def last(self):
        return self

    def nth(self, _i):
        return self

    async def count(self):
        return self._count

    async def is_visible(self):
        return self._visible

    async def wait_for(self, state=None, timeout=None):
        if not self._visible:
            raise TimeoutError(f"{self.key} never became {state}")

    async def click(self, **kwargs):
        self.page.actions.append(("click", self.key))
        handler = self.page.on_click.get(self.key)
        if handler:
            handler()

    async def hover(self, **kwargs):
        self.page.actions.append(("hover", self.key))
        handler = self.page.on_hover.get(self.key)
        if handler:
            handler()

    async def inner_text(self):
        return self._text

    async def input_value(self):
        return self._value

    async def fill(self, value):
        self._value = value
        self.page.actions.append(("fill", self.key, value))

    async def press(self, key):
        self.page.actions.append(("press", self.key, key))

    async def set_input_files(self, paths):
        self.page.actions.append(("set_input_files", self.key, paths))
        handler = getattr(self, "on_set_input_files", None)
        if handler:
            handler(paths)


class FakeFileChooser:
    def __init__(self, page):
        self.page = page

    async def set_files(self, paths):
        self.page.chooser_files = [paths] if isinstance(paths, str) else list(paths)
        if self.page.file_chooser_sets_url:
            self.page.url = self.page.file_chooser_sets_url


class FakeFileChooserContext:
    def __init__(self, page):
        self.page = page

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def value(self):
        async def _value():
            return FakeFileChooser(self.page)

        return _value()


class FakeKeyboard:
    def __init__(self, page):
        self.page = page

    async def press(self, key):
        self.page.actions.append(("keyboard", key))


class FakeCDP:
    def __init__(self, page):
        self.page = page

    async def send(self, method, params=None):
        self.page.cdp_calls.append((method, params or {}))
        return {}


class FakeContext:
    def __init__(self, page):
        self.page = page

    async def new_cdp_session(self, _page):
        return FakeCDP(self.page)


class FakePage:
    def __init__(self, url="https://colab.research.google.com/drive/abc"):
        self.url = url
        self.actions = []
        self.cdp_calls = []
        self.locators = {}
        self.on_click = {}
        self.on_hover = {}
        self.chooser_files = None
        self.file_chooser_sets_url = None
        self.keyboard = FakeKeyboard(self)
        self.context = FakeContext(self)

    def expect_file_chooser(self, timeout=None):
        return FakeFileChooserContext(self)

    def add(self, key, **kwargs):
        self.locators[key] = FakeLocator(self, key, **kwargs)
        return self.locators[key]

    def _lookup(self, key):
        if key in self.locators:
            return self.locators[key]
        return FakeLocator(self, key, count=0, visible=False)

    def get_by_text(self, text, exact=False):
        return self._lookup(("text", text))

    def get_by_role(self, role, name=None, exact=False):
        return self._lookup(("role", role, name))

    def locator(self, selector):
        return self._lookup(("css", selector))

    async def title(self):
        return "nb - Colab"

    async def screenshot(self, path=None):
        return path


@pytest.fixture
def page():
    return FakePage()


@pytest.fixture
def colab(page):
    b = ColabBrowser()
    b.page = page
    return b


# --------------------------------------------------------------------------
# runtime control
# --------------------------------------------------------------------------


class TestInterruptExecution:
    @pytest.mark.asyncio
    async def test_uses_the_runtime_menu(self, colab, page):
        page.add(("text", "Runtime"))
        page.add(("text", "Interrupt execution"))

        assert await colab.interrupt_execution() is True
        assert ("click", ("text", "Runtime")) in page.actions
        assert ("click", ("text", "Interrupt execution")) in page.actions

    @pytest.mark.asyncio
    async def test_falls_back_to_the_keyboard_shortcut(self, colab, page):
        """The menu entry is disabled when Colab thinks nothing is running."""
        page.add(("text", "Runtime"))  # menu opens, but no Interrupt entry

        assert await colab.interrupt_execution() is True
        assert ("keyboard", "Control+m") in page.actions
        assert ("keyboard", "i") in page.actions

    @pytest.mark.asyncio
    async def test_no_page(self):
        assert await ColabBrowser().interrupt_execution() is False

    @pytest.mark.asyncio
    async def test_does_not_restart_the_runtime(self, colab, page):
        """Interrupting must never touch the entries that discard kernel state."""
        page.add(("text", "Runtime"))
        page.add(("text", "Interrupt execution"))

        await colab.interrupt_execution()

        clicked = {a[1] for a in page.actions if a[0] == "click"}
        assert ("text", "Restart session") not in clicked
        assert ("text", "Disconnect and delete runtime") not in clicked


class TestRestartKernel:
    @pytest.mark.asyncio
    async def test_restarts_and_confirms(self, colab, page):
        page.add(("text", "Runtime"))
        page.add(("text", "Restart session"))
        page.add(("text", "Yes"))

        assert await colab.restart_kernel() is True
        assert ("click", ("text", "Restart session")) in page.actions
        assert ("click", ("text", "Yes")) in page.actions

    @pytest.mark.asyncio
    async def test_accepts_the_older_menu_wording(self, colab, page):
        page.add(("text", "Runtime"))
        page.add(("text", "Restart runtime"))

        assert await colab.restart_kernel() is True
        assert ("click", ("text", "Restart runtime")) in page.actions

    @pytest.mark.asyncio
    async def test_reports_failure_when_the_entry_is_absent(self, colab, page):
        page.add(("text", "Runtime"))
        assert await colab.restart_kernel() is False

    @pytest.mark.asyncio
    async def test_no_page(self):
        assert await ColabBrowser().restart_kernel() is False


class TestDeleteRuntime:
    @pytest.mark.asyncio
    async def test_deletes_and_confirms(self, colab, page):
        page.add(("text", "Runtime"))
        page.add(("text", "Disconnect and delete runtime"))
        page.add(("text", "Yes"))

        assert await colab.delete_runtime() is True
        assert ("click", ("text", "Disconnect and delete runtime")) in page.actions
        assert ("click", ("text", "Yes")) in page.actions

    @pytest.mark.asyncio
    async def test_accepts_the_shorter_menu_wording(self, colab, page):
        page.add(("text", "Runtime"))
        page.add(("text", "Delete runtime"))

        assert await colab.delete_runtime() is True
        assert ("click", ("text", "Delete runtime")) in page.actions

    @pytest.mark.asyncio
    async def test_succeeds_when_no_confirmation_is_shown(self, colab, page):
        # Colab skips the prompt when the runtime holds nothing worth losing.
        page.add(("text", "Runtime"))
        page.add(("text", "Disconnect and delete runtime"))

        assert await colab.delete_runtime() is True

    @pytest.mark.asyncio
    async def test_reports_failure_when_the_entry_is_absent(self, colab, page):
        page.add(("text", "Runtime"))
        assert await colab.delete_runtime() is False

    @pytest.mark.asyncio
    async def test_does_not_settle_for_a_mere_restart(self, colab, page):
        """Deleting must never silently degrade into restarting: the caller asked
        for the VM to go away, and a restart would leave it running and billing."""
        page.add(("text", "Runtime"))
        page.add(("text", "Restart session"))

        assert await colab.delete_runtime() is False
        assert ("click", ("text", "Restart session")) not in page.actions

    @pytest.mark.asyncio
    async def test_no_page(self):
        assert await ColabBrowser().delete_runtime() is False


# --------------------------------------------------------------------------
# save / export
# --------------------------------------------------------------------------


class TestSaveNotebook:
    @pytest.mark.asyncio
    async def test_uses_the_file_menu_not_the_keyboard(self, colab, page):
        """Ctrl+S silently does nothing when the editor lacks keyboard focus."""
        page.add(("text", "File"))
        page.add(("role", "menuitem", "Save"))

        assert await colab.save_notebook() is True
        assert ("click", ("role", "menuitem", "Save")) in page.actions
        assert not any(a[0] == "keyboard" for a in page.actions)

    @pytest.mark.asyncio
    async def test_reports_failure_and_closes_the_menu(self, colab, page):
        page.add(("text", "File"))

        assert await colab.save_notebook() is False
        assert ("keyboard", "Escape") in page.actions

    @pytest.mark.asyncio
    async def test_no_page(self):
        assert await ColabBrowser().save_notebook() is False


class TestDownloadNotebook:
    def _menu(self, page):
        page.add(("text", "File"))
        # The parent entry's text is "Download ►", so it is matched by
        # accessible-name substring rather than exact text.
        page.add(("role", "menuitem", "Download"))
        page.add(("role", "menuitem", "Download .ipynb"))

    @pytest.mark.asyncio
    async def test_hovers_the_submenu_then_clicks_ipynb(
        self, colab, page, tmp_path, monkeypatch
    ):
        downloads = tmp_path / "dl"
        monkeypatch.setattr(browser_mod, "DOWNLOAD_DIR", str(downloads))
        self._menu(page)
        page.on_click[("role", "menuitem", "Download .ipynb")] = lambda: (
            downloads.mkdir(exist_ok=True),
            (downloads / "nb.ipynb").write_text(json.dumps({"cells": []})),
        )
        dest = tmp_path / "out.ipynb"

        result = await colab.download_notebook(str(dest), timeout_ms=5000)

        assert result == str(dest)
        assert ("hover", ("role", "menuitem", "Download")) in page.actions
        assert ("click", ("role", "menuitem", "Download .ipynb")) in page.actions
        assert json.loads(dest.read_text()) == {"cells": []}

    @pytest.mark.asyncio
    async def test_points_chrome_at_a_snap_writable_directory(
        self, colab, page, tmp_path, monkeypatch
    ):
        """Snap chromium cannot write to a host /tmp path, so the download
        directory must be set explicitly over CDP."""
        downloads = tmp_path / "dl"
        monkeypatch.setattr(browser_mod, "DOWNLOAD_DIR", str(downloads))
        self._menu(page)
        page.on_click[("role", "menuitem", "Download .ipynb")] = lambda: (
            downloads.mkdir(exist_ok=True),
            (downloads / "nb.ipynb").write_text("{}"),
        )

        await colab.download_notebook(str(tmp_path / "out.ipynb"), timeout_ms=5000)

        method, params = page.cdp_calls[0]
        assert method == "Browser.setDownloadBehavior"
        assert params["behavior"] == "allow"
        assert params["downloadPath"] == str(downloads)

    @pytest.mark.asyncio
    async def test_rejects_an_empty_download(self, colab, page, tmp_path, monkeypatch):
        downloads = tmp_path / "dl"
        monkeypatch.setattr(browser_mod, "DOWNLOAD_DIR", str(downloads))
        self._menu(page)
        page.on_click[("role", "menuitem", "Download .ipynb")] = lambda: (
            downloads.mkdir(exist_ok=True),
            (downloads / "nb.ipynb").write_bytes(b""),
        )

        with pytest.raises(RuntimeError, match="empty"):
            await colab.download_notebook(str(tmp_path / "out.ipynb"), timeout_ms=1200)

    @pytest.mark.asyncio
    async def test_times_out_when_nothing_downloads(
        self, colab, page, tmp_path, monkeypatch
    ):
        downloads = tmp_path / "dl"
        downloads.mkdir()
        monkeypatch.setattr(browser_mod, "DOWNLOAD_DIR", str(downloads))
        self._menu(page)

        with pytest.raises(RuntimeError, match="no download appeared"):
            await colab.download_notebook(str(tmp_path / "out.ipynb"), timeout_ms=1500)

    @pytest.mark.asyncio
    async def test_no_page(self, tmp_path):
        with pytest.raises(RuntimeError, match="browser not open"):
            await ColabBrowser().download_notebook(str(tmp_path / "x.ipynb"))


class TestSettleDownload:
    @pytest.mark.asyncio
    async def test_ignores_the_in_progress_marker(self, colab, tmp_path):
        directory = tmp_path / "dl"
        directory.mkdir()
        (directory / "nb.ipynb.crdownload").write_text("partial")

        async def finish():
            await asyncio.sleep(0.6)
            (directory / "nb.ipynb").write_text("done")

        task = asyncio.create_task(finish())
        found = await colab._settle_download(str(directory), set(), 5)
        await task

        assert found.endswith("nb.ipynb")

    @pytest.mark.asyncio
    async def test_ignores_files_that_were_already_there(self, colab, tmp_path):
        directory = tmp_path / "dl"
        directory.mkdir()
        (directory / "old.ipynb").write_text("old")
        before = set(os.listdir(directory))

        async def finish():
            await asyncio.sleep(0.4)
            (directory / "new.ipynb").write_text("new")

        task = asyncio.create_task(finish())
        found = await colab._settle_download(str(directory), before, 5)
        await task

        assert found.endswith("new.ipynb")


# --------------------------------------------------------------------------
# sign-in
# --------------------------------------------------------------------------


class TestProfileEmail:
    def test_reads_the_account_from_the_chrome_profile(self, tmp_path, monkeypatch):
        default = tmp_path / "Default"
        default.mkdir()
        (default / "Preferences").write_text(
            json.dumps({"account_info": [{"email": "someone@example.test"}]})
        )
        monkeypatch.setattr(browser_mod, "SRC_PROFILE", str(tmp_path))
        monkeypatch.setattr(browser_mod, "GOOGLE_EMAIL", "")

        assert browser_mod._profile_email() == "someone@example.test"

    def test_the_env_override_wins(self, monkeypatch):
        monkeypatch.setattr(browser_mod, "GOOGLE_EMAIL", "override@example.test")
        assert browser_mod._profile_email() == "override@example.test"

    def test_missing_profile_is_not_fatal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(browser_mod, "SRC_PROFILE", str(tmp_path / "nope"))
        monkeypatch.setattr(browser_mod, "GOOGLE_EMAIL", "")
        assert browser_mod._profile_email() == ""


class TestEnsureSignedIn:
    @pytest.mark.asyncio
    async def test_no_op_when_already_signed_in(self, colab, page):
        assert await colab.ensure_signed_in() is True
        assert page.actions == []

    @pytest.mark.asyncio
    async def test_clicks_through_colabs_own_interstitial(self, colab, page):
        """Colab renders "You must be logged in..." rather than redirecting."""
        page.add(("text", "must be logged in with a Google Account"))
        page.add(("text", "Sign in"))
        signin_clicked = []

        def on_click():
            signin_clicked.append(True)
            # the click takes us to the account flow, and then back signed in
            page.locators.pop(("text", "must be logged in with a Google Account"))

        page.on_click[("text", "Sign in")] = on_click

        assert await colab.ensure_signed_in(timeout_s=5) is True
        assert signin_clicked

    @pytest.mark.asyncio
    async def test_lets_chrome_autofill_the_saved_password(self, colab, page, monkeypatch):
        """colab-mcp never types a password: it clicks the field so Chrome's own
        password manager fills it, then submits."""
        page.url = "https://accounts.google.com/signin/v2/challenge/pwd"
        monkeypatch.setattr(browser_mod, "GOOGLE_EMAIL", "")
        monkeypatch.setattr(browser_mod, "_profile_email", lambda: "")
        # Chrome has already autofilled the field.
        page.add(("css", "input[type=password]"), value="autofilled-by-chrome")

        def submitted():
            page.url = "https://colab.research.google.com/drive/abc"

        page.on_click[("css", "input[type=password]")] = lambda: None
        original_press = FakeLocator.press

        async def press(self, key):
            await original_press(self, key)
            if key == "Enter":
                submitted()

        monkeypatch.setattr(FakeLocator, "press", press)

        assert await colab.ensure_signed_in(timeout_s=10) is True
        assert ("press", ("css", "input[type=password]"), "Enter") in page.actions
        # nothing was typed into the password field
        assert not any(
            a[0] == "fill" and a[1] == ("css", "input[type=password]") for a in page.actions
        )

    @pytest.mark.asyncio
    async def test_gives_up_when_chrome_does_not_autofill(self, colab, page, monkeypatch):
        page.url = "https://accounts.google.com/signin/v2/challenge/pwd"
        monkeypatch.setattr(browser_mod, "_profile_email", lambda: "")
        page.add(("css", "input[type=password]"), value="")  # no saved credential

        assert await colab.ensure_signed_in(timeout_s=5) is False

    @pytest.mark.asyncio
    async def test_fills_the_email_step(self, colab, page, monkeypatch):
        page.url = "https://accounts.google.com/signin/v2/identifier"
        monkeypatch.setattr(browser_mod, "_profile_email", lambda: "user@example.test")
        page.add(("css", "input[type=email]"), value="")
        page.add(("role", "button", "Next"))

        def on_next():
            page.url = "https://colab.research.google.com/drive/abc"

        page.on_click[("role", "button", "Next")] = on_next

        assert await colab.ensure_signed_in(timeout_s=10) is True
        assert ("fill", ("css", "input[type=email]"), "user@example.test") in page.actions

    @pytest.mark.asyncio
    async def test_picks_the_profiles_account_from_the_chooser(
        self, colab, page, monkeypatch
    ):
        page.url = "https://accounts.google.com/signin/v2/chooser"
        monkeypatch.setattr(browser_mod, "_profile_email", lambda: "user@example.test")
        page.add(("text", "user@example.test"))

        def on_pick():
            page.url = "https://colab.research.google.com/drive/abc"

        page.on_click[("text", "user@example.test")] = on_pick

        assert await colab.ensure_signed_in(timeout_s=10) is True
        assert ("click", ("text", "user@example.test")) in page.actions

    @pytest.mark.asyncio
    async def test_times_out_without_hanging(self, colab, page, monkeypatch):
        page.url = "https://accounts.google.com/signin/v2/identifier"
        monkeypatch.setattr(browser_mod, "_profile_email", lambda: "")

        assert await colab.ensure_signed_in(timeout_s=2) is False

    @pytest.mark.asyncio
    async def test_no_page(self):
        assert await ColabBrowser().ensure_signed_in() is False


class TestPreservedBehaviour:
    """The tools the improvements doc asked us not to regress."""

    @pytest.mark.asyncio
    async def test_click_connect_ignores_the_runtime_connect_button(self, colab, page):
        """Only the MCP authorization dialog may be clicked, never the toolbar's
        unrelated runtime Connect button."""
        page.add(("role", "button", "Connect"))  # toolbar button present...
        # ...but the authorization dialog text is not.

        assert await colab.click_connect() is False
        assert page.actions == []

    @pytest.mark.asyncio
    async def test_click_connect_accepts_the_authorization_dialog(self, colab, page):
        page.add(("text", "local Colab MCP server"))
        page.add(("role", "button", "Connect"))

        assert await colab.click_connect() is True
        assert ("click", ("role", "button", "Connect")) in page.actions

    @pytest.mark.asyncio
    async def test_set_runtime_accelerator_still_saves(self, colab, page):
        page.add(("text", "Runtime"))
        page.add(("text", "Change runtime type"))
        page.add(("text", "A100 GPU"))
        page.add(("text", "Save"))

        result = await colab.set_runtime_accelerator("A100 GPU")

        assert "A100 GPU" in result
        assert ("click", ("text", "A100 GPU")) in page.actions
        assert ("click", ("text", "Save")) in page.actions


class TestDriveIdFromUrl:
    def test_extracts_the_id(self):
        assert (
            browser_mod.drive_id_from_url(
                "https://colab.research.google.com/drive/1pIwBb-TQ_M3q#mcpProxyToken=x"
            )
            == "1pIwBb-TQ_M3q"
        )

    def test_scratch_and_blank_urls_have_none(self):
        assert browser_mod.drive_id_from_url(
            "https://colab.research.google.com/notebooks/empty.ipynb"
        ) == ""
        assert browser_mod.drive_id_from_url("") == ""
        assert browser_mod.drive_id_from_url(None) == ""


class TestUploadNotebook:
    def _menu(self, page):
        # Start off a /drive/ URL, so "we landed on the uploaded notebook" is a
        # real assertion rather than one the starting URL already satisfies.
        page.url = "https://colab.research.google.com/notebooks/empty.ipynb"
        page.add(("text", "File"))
        page.add(("role", "menuitem", "Upload notebook"))

    @pytest.mark.asyncio
    async def test_drives_the_hidden_file_input(self, colab, page, tmp_path):
        """set_input_files avoids raising an OS file chooser we cannot drive."""
        src = tmp_path / "nb.ipynb"
        src.write_text(json.dumps({"cells": []}))
        self._menu(page)
        file_input = page.add(("css", "input[type=file]"))

        def on_set(_paths):
            page.url = "https://colab.research.google.com/drive/NEWID123"

        file_input.on_set_input_files = on_set

        assert await colab.upload_notebook(str(src), timeout_s=5) == "NEWID123"
        assert ("set_input_files", ("css", "input[type=file]"), str(src)) in page.actions

    @pytest.mark.asyncio
    async def test_falls_back_to_the_file_chooser(self, colab, page, tmp_path):
        src = tmp_path / "nb.ipynb"
        src.write_text(json.dumps({"cells": []}))
        self._menu(page)
        page.add(("text", "Browse"))
        page.file_chooser_sets_url = "https://colab.research.google.com/drive/VIACHOOSER"

        assert await colab.upload_notebook(str(src), timeout_s=5) == "VIACHOOSER"
        assert page.chooser_files == [str(src)]

    @pytest.mark.asyncio
    async def test_does_not_report_the_notebook_we_started_on(self, colab, page, tmp_path):
        """Regression: uploading from an already-open /drive/ notebook made the
        poll match the current URL instantly and return the OLD id, so the
        bridge was re-attached to the wrong notebook."""
        src = tmp_path / "nb.ipynb"
        src.write_text(json.dumps({"cells": []}))
        page.url = "https://colab.research.google.com/drive/OLDID"
        page.add(("text", "File"))
        page.add(("role", "menuitem", "Upload notebook"))
        file_input = page.add(("css", "input[type=file]"))

        navigated = []

        def on_set(_paths):
            # Colab takes a moment to upload before it navigates.
            navigated.append(True)

        file_input.on_set_input_files = on_set

        async def land_later():
            await asyncio.sleep(1.5)
            page.url = "https://colab.research.google.com/drive/NEWID"

        task = asyncio.create_task(land_later())
        result = await colab.upload_notebook(str(src), timeout_s=10)
        await task

        assert result == "NEWID"

    @pytest.mark.asyncio
    async def test_rejects_a_missing_local_file(self, colab, page, tmp_path):
        with pytest.raises(RuntimeError, match="no such file"):
            await colab.upload_notebook(str(tmp_path / "absent.ipynb"))

    @pytest.mark.asyncio
    async def test_times_out_when_colab_never_navigates(self, colab, page, tmp_path):
        src = tmp_path / "nb.ipynb"
        src.write_text("{}")
        self._menu(page)
        page.add(("css", "input[type=file]"))  # upload accepted, but no navigation

        with pytest.raises(RuntimeError, match="did not open an uploaded notebook"):
            await colab.upload_notebook(str(src), timeout_s=2)

    @pytest.mark.asyncio
    async def test_no_page(self, tmp_path):
        src = tmp_path / "nb.ipynb"
        src.write_text("{}")
        with pytest.raises(RuntimeError, match="browser not open"):
            await ColabBrowser().upload_notebook(str(src))


class TestSignedOutDetection:
    """A Colab notebook that renders anonymously shows no interstitial, only a
    "Sign in" button in the toolbar. Missing it made ensure_signed_in() a no-op
    and let a signed-out session report itself as connected."""

    @pytest.mark.asyncio
    async def test_detects_toolbar_sign_in_button(self, colab, page):
        page.add(("role", "button", "Sign in"))
        assert await colab._needs_signin_on_colab() is True
        assert await colab.is_signed_in() is False

    @pytest.mark.asyncio
    async def test_detects_sign_in_link(self, colab, page):
        page.add(("role", "link", "Sign in"))
        assert await colab._needs_signin_on_colab() is True

    @pytest.mark.asyncio
    async def test_still_detects_the_interstitial(self, colab, page):
        page.add(("text", "must be logged in with a Google Account"))
        assert await colab._needs_signin_on_colab() is True

    @pytest.mark.asyncio
    async def test_signed_in_page_has_no_affordance(self, colab, page):
        assert await colab._needs_signin_on_colab() is False
        assert await colab.is_signed_in() is True

    @pytest.mark.asyncio
    async def test_ignores_sign_in_text_off_colab(self, colab, page):
        # Google's own sign-in pages carry a "Sign in" button; treating that as
        # "Colab needs sign-in" would bounce us around the flow forever.
        page.url = "https://accounts.google.com/v3/signin/challenge/pwd"
        page.add(("role", "button", "Sign in"))
        assert await colab._needs_signin_on_colab() is False

    @pytest.mark.asyncio
    async def test_on_signin_page_is_not_signed_in(self, colab, page):
        page.url = "https://accounts.google.com/v3/signin/accountchooser"
        assert await colab.is_signed_in() is False


class TestProfileReclaim:
    """A server that dies without closing its browser strands a chromium on the
    profile, and the next launch fails on a lock nobody is left to clear. These
    use real processes: the bug was that terminate() was never waited on, which
    a mock would have happily reported as working."""

    @staticmethod
    def _spawn(profile, *extra):
        import subprocess
        import sys

        return subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)",
             f"--user-data-dir={profile}", *extra]
        )

    def test_finds_the_process_holding_the_profile(self, tmp_path):
        profile = str(tmp_path / "prof")
        proc = self._spawn(profile)
        try:
            assert proc.pid in browser_mod._chrome_pids_using_profile(profile)
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_ignores_helper_processes(self, tmp_path):
        """Only the browser process holds the lock; killing renderers is noise."""
        profile = str(tmp_path / "prof")
        proc = self._spawn(profile, "--type=renderer")
        try:
            assert proc.pid not in browser_mod._chrome_pids_using_profile(profile)
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_ignores_a_different_profile(self, tmp_path):
        proc = self._spawn(str(tmp_path / "other"))
        try:
            assert browser_mod._chrome_pids_using_profile(str(tmp_path / "mine")) == []
        finally:
            proc.kill()
            proc.wait(timeout=10)

    def test_terminate_stops_the_process(self, tmp_path):
        proc = self._spawn(str(tmp_path / "prof"))
        try:
            assert browser_mod._terminate_pids([proc.pid], timeout_s=10) == []
            assert not browser_mod._pid_alive(proc.pid) or proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=10)

    def test_kills_what_ignores_sigterm(self, tmp_path):
        """Chrome can decline SIGTERM. The profile still has to come free."""
        import subprocess
        import sys

        profile = str(tmp_path / "prof")
        proc = subprocess.Popen(
            [sys.executable, "-c",
             "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
             f"--user-data-dir={profile}"]
        )
        try:
            # Give it a moment to install the handler before signalling.
            time.sleep(0.5)
            assert browser_mod._terminate_pids([proc.pid], timeout_s=1.5) == [proc.pid]
            proc.wait(timeout=10)
            assert proc.poll() is not None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=10)

    def test_survives_a_pid_that_is_already_gone(self, tmp_path):
        proc = self._spawn(str(tmp_path / "prof"))
        pid = proc.pid
        proc.kill()
        proc.wait(timeout=10)
        # Reaped mid-flight is the normal race, not an error.
        assert browser_mod._terminate_pids([pid], timeout_s=1) == []


class TestProfileSyncOptOut:
    """COLAB_MCP_SKIP_PROFILE_SYNC keeps a signed-in automation profile intact,
    which is what makes unattended use possible."""

    def test_sync_copies_by_default(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        (src / "Default").mkdir(parents=True)
        (src / "Default" / "Cookies").write_text("cookie-data")
        dst = tmp_path / "pw"
        monkeypatch.setattr(browser_mod, "SRC_PROFILE", str(src))
        monkeypatch.setattr(browser_mod, "PW_PROFILE", str(dst))
        monkeypatch.setattr(browser_mod, "SKIP_PROFILE_SYNC", False)

        browser_mod._sync_login_profile()

        assert (dst / "Default" / "Cookies").read_text() == "cookie-data"

    def test_sync_is_a_no_op_when_opted_out(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        (src / "Default").mkdir(parents=True)
        (src / "Default" / "Cookies").write_text("signed-out-source")
        dst = tmp_path / "pw"
        (dst / "Default").mkdir(parents=True)
        (dst / "Default" / "Cookies").write_text("signed-in-automation")
        monkeypatch.setattr(browser_mod, "SRC_PROFILE", str(src))
        monkeypatch.setattr(browser_mod, "PW_PROFILE", str(dst))
        monkeypatch.setattr(browser_mod, "SKIP_PROFILE_SYNC", True)

        browser_mod._sync_login_profile()

        # The automation profile's own login survives untouched.
        assert (dst / "Default" / "Cookies").read_text() == "signed-in-automation"
