# ⚠️ URGENT — MUST FIX: a Colab dialog can deadlock every UI-driven tool

*Filed 2026-09-21 after an 8-hour unattended run (vLLM generation → HF upload → runtime teardown).
Severity: the server keeps reporting healthy while the user is unable to stop paying for a VM.*

> **Status: fixed on `local-patches`, 2026-09-21.** Kept below as the bug record.
>
> 1. Every menu-driven path (`set_gpu_runtime`, `restart_kernel`, `delete_runtime`, the File menu
>    behind `export_notebook`/`import_notebook`) now clears open `mwc-dialog` overlays first, by
>    clicking `[dialogaction="cancel"]` from JS (the buttons are `md-text-button`; `inert` dialogs
>    are unreachable to Playwright locators). Stacked prompts are cleared in one pass. A dialog that
>    will not close raises `BlockingDialogError` with the dialog's text, instead of a Playwright
>    timeout. The MCP bridge's own "Connect to a local Colab MCP server" prompt is never cancelled.
>    The secrets prompt itself is answered **"Grant access"** rather than cancelled: a background
>    watcher polls the page every second while the browser is open and clicks it while the kernel's
>    `userdata.get()` is still waiting, so the call succeeds and nothing is left to block a menu.
>    Set `COLAB_MCP_GRANT_SECRETS=0` to cancel instead (auto-granting hands the account's secrets to
>    whatever notebook the agent drives). Every other dialog is still cancelled.
>    New tool **`dismiss_colab_dialog(dismiss=true, grant_secrets=null)`** is the escape hatch: it
>    lists open dialogs, closes them, and returns `granted_secrets` / `pending_secrets` parsed from
>    any `user-secrets-prompt`, so the agent can ask the user to grant access rather than retry.
> 2. `browser.delete_runtime` / `restart_kernel` return `False` only when the Runtime menu opened and
>    had no matching entry. Any other failure propagates, and the kernel layer reports it as
>    `error` (first line plus the "intercepts pointer events" line) with a note that the runtime
>    was **not** deleted, instead of "is a runtime connected?".
> 3. `get_runtime_info` first reads the toolbar's connect button (`colab-connect-button` shadow DOM:
>    "Connect"/"Reconnect" means nothing attached) and remembers a successful `delete_runtime`
>    until the next execution. When nothing is attached it returns `runtime_attached: false` without
>    running a cell, so it no longer provisions a VM.
> 4. `MAX_TRANSFER_BYTES` defaults to 8 GiB and is overridable with `COLAB_MCP_MAX_TRANSFER_BYTES`;
>    the cap is stated in the `download_file`/`upload_file` descriptions and in the error, and
>    progress is logged every 16 MB.
>
> Restart Claude Code (or whichever client launched the server) to pick the change up.

## 1. `user-secrets-prompt` blocks every pointer event, and nothing can dismiss it

**What happens.** Calling `google.colab.userdata.get("SOME_SECRET")` from a notebook that has not
been granted access to that secret raises `TimeoutException` *and* leaves this in the top-level DOM:

```html
<mwc-dialog class="user-secrets-prompt" open inert aria-hidden="true">
  <p>Notebook titled 'x.ipynb' does not have access to secret named 'SOME_SECRET'. Grant access?</p>
  <md-text-button slot="secondaryAction" dialogaction="cancel">Cancel</md-text-button>
  <md-text-button slot="primaryAction"   dialogaction="ok">Grant access</md-text-button>
</mwc-dialog>
```

That dialog intercepts pointer events for the whole page. Every tool that drives the menu bar then
fails with a Playwright timeout:

```
Locator.click: Timeout 30000ms exceeded.
  - waiting for get_by_text("Runtime", exact=True).first
  - <mwc-dialog open="" class="user-secrets-prompt">…</mwc-dialog> intercepts pointer events
```

**Blast radius.** Everything routed through `browser._runtime_menu`, `browser.set_runtime_accelerator`
and `browser._file_menu` is dead: `set_gpu_runtime`, `delete_runtime`, `restart_kernel`.

**Why it is urgent, not cosmetic.** `browser.delete_runtime`'s own docstring says it "is also the only
way to stop a session from burning compute units while idle". So the one dialog a misconfigured
`userdata.get` raises is enough to take away the user's only means of stopping the meter — while the
kernel stays fully reachable, so `execute_transient`, `run_code_cell_async`, `upload_file` and
`download_file` all keep succeeding and the server looks fine.

**It survives a reload.** Re-opening the same notebook id via `open_colab_notebook` does not clear it.
One instance is raised per failed `userdata.get`, so repeated calls stack them.

**There is no escape hatch in the tool surface.** The MCP exposes no click/press/keyboard primitive,
so an agent cannot dismiss a dialog it did not expect. Recovery required attaching a *separate*
Playwright client to the browser's CDP port and calling
`document.querySelector('mwc-dialog.user-secrets-prompt md-text-button[dialogaction="cancel"]').click()`.
An end user on a remote/headless box has no way to do that.

**Note for whoever fixes it:** the buttons are `md-text-button`, *not* `mwc-button`. A selector written
against `mwc-button` silently matches nothing inside this dialog.

### Suggested fix
1. Before any menu interaction, dismiss (or fail loudly on) blocking overlays — query
   `mwc-dialog[open]` and click `md-text-button[dialogaction="cancel"]`.
2. Expose a `dismiss_dialog` tool, or a generic `click_text` escape hatch.
3. Better: surface the pending-secret state as a tool result so the agent can ask the user to grant
   access *before* anything is wedged.

## 2. `delete_runtime` misreports an overlay as a missing menu entry

`browser.delete_runtime` wraps `_runtime_menu` in `contextlib.suppress(Exception)` inside a
`for … else`. Any failure — including the pointer interception above — skips the `break`, falls to the
`else`, and returns:

```
"could not find the delete-runtime entry; is a runtime connected?"
```

The runtime *was* connected; the menu was simply unclickable. That message sent the caller debugging
runtime connectivity for a DOM problem. **Do not swallow the exception — log it and report the real
cause** (`kernel.py` builds the user-facing string around this same false branch).

## 3. `get_runtime_info` silently resurrects a VM you just deleted

`kernel.runtime_info` executes a payload on the VM via `_run_payload`. Calling it after a successful
`delete_runtime` therefore *starts a brand-new runtime*:

```
delete_runtime  -> {"deleted": true, "note": "…The next execution starts a fresh runtime."}
get_runtime_info-> hostname e73e3b5f9e1b → d1b53a75182a, vram_used 0, torch 2.13.0+cu130 → 2.11.0+cu128
```

A call that reads as a read-only status check allocates hardware and restarts billing. Either answer
`get_runtime_info` from cached/frontend state when no runtime is attached, or have it report
`connected: false` without provisioning.

## 4. Minor: 256 MB transfer ceiling with no chunked path

`MAX_TRANSFER_BYTES = 256 * 1024 * 1024` (`kernel.py:62`) hard-fails `download_file`:

```
/content/features/X_005.npy is 412876928 bytes; limit is 268435456
```

`FILE_CHUNK_BYTES` already exists, so the transport chunks internally; only the whole-file guard
blocks it. Pulling a 7.8 GB activation dump meant splitting it on the VM by hand. Please allow
multi-part transfers, or document the cap in the tool description so agents plan around it.

---

# Colab-mcp

An MCP server for bridging your local agent to a Colab session in the browser.

# Supported Clients
This MCP server requires client support for `notifications/tools/list_changed` and for the client to be running locally on your device. 

Popular clients that fit these criteria include:
- Gemini CLI
- Claude Code
- Windsurf


# Setup

- Install `uv` (`pip install uv`)
- Configure for usage (eg for mcp.json style services):

```
...
  "mcpServers": {
    "colab-mcp": {
      "command": "uvx",
      "args": ["git+https://github.com/googlecolab/colab-mcp"],
      "timeout": 30000
    }
  }
...
```

(If you have a non-standard default package index (**Googlers**), you may also need to add `--index https://pypi.org/simple`)

# Issues & Discussions

We are using GitHub [discussions](https://github.com/googlecolab/colab-mcp/discussions) as the
place for issue discussion and feature requests. As discussions mature into action items, we
will add those items as issues. This helps us ensure that issues in the issue tracker are
well-understood, deduplicated, and actionable. For these reasons, **please do <u>NOT</u> open
issues directly.** 

# Contributing 
We unfortunately don't have the bandwidth to support review of external contributions, and we 
don't want user PRs to languish, so we aren't accepting any external contributions right now.

If you have a great idea or pain point, we would love to hear about it on our 
[discussions](https://github.com/googlecolab/colab-mcp/discussions) page - the preferred place 
for issue discussion and feature requests.

# Internal - For Colab Developers

### Prerequisites

- `uv` is required (`pip install uv`)
- Configure git hooks to run repo presubmits

```shell
git config core.hooksPath .githooks
```

### Gemini CLI setup

```
...
  "mcpServers": {
    "colab-mcp": {
      "command": "uv",
      "args": ["run", "colab-mcp"],
      "cwd": "/path/to/github/colab-mcp",
      "timeout": 30000
    }
  }
...
```
