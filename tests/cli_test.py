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

from colab_mcp import DEFAULT_LOG_DIR, parse_args


def test_enable_proxy_default():
    assert parse_args([]).enable_proxy is True


def test_enable_proxy_explicit():
    assert parse_args(["--enable-proxy"]).enable_proxy is True


def test_disable_proxy():
    """Regression: the old store_true/default=True flag could never be off."""
    assert parse_args(["--no-enable-proxy"]).enable_proxy is False


def test_log_default_is_stable():
    """Regression: the old default minted a fresh mkdtemp dir on every parse."""
    assert parse_args([]).log == DEFAULT_LOG_DIR
    assert parse_args([]).log == DEFAULT_LOG_DIR


def test_log_override():
    assert parse_args(["-l", "/some/dir"]).log == "/some/dir"
