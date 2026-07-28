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

import pytest

from colab_mcp import browser as browser_mod


@pytest.fixture(autouse=True)
def fast_ui_pauses(monkeypatch):
    """Collapse the browser's UI settle waits.

    They exist because the next element does not exist yet to be waited on, so
    they are real in production — but running them at full speed would make the
    suite spend most of its time asleep.
    """
    monkeypatch.setattr(browser_mod, "UI_PAUSE_SCALE", 0.01)
