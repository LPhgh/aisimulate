# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Defect canary: AFD layer measurements do not survive a ColumnsAFD round-trip.

``AFDInferenceSession`` writes ``afd_layer_measurements`` into its result dict, then builds a
DataFrame projected onto ``common.ColumnsAFD``. That schema omits the key, so any path that
rebuilds the result dict from the frame drops it. One such path is reachable today:
``cli_estimate`` always constructs a ``SpeculativeDecodingProfile``, and with ``nextn > 0`` and
``expected_accepted_tokens > 0`` the decode/static projection in
``aisimulate/sdk/speculative.py`` calls ``set_result_dict(frame.iloc[0].to_dict())``.

These tests pin the two halves of that coupling without importing the estimator, so they run
without the native runtime. They are a canary, not a specification: when the key is added to
``ColumnsAFD`` (or the projection starts preserving extra keys) the first test fails on purpose.
Retire it then, together with the AFD+MTP limitation noted in tests/e2e/README_afd_estimate.md.
"""

from __future__ import annotations

from pathlib import Path

from aisimulate_core.sdk import common

_SRC = Path(common.__file__).resolve().parents[2]
_SESSION = _SRC / "aisimulate/sdk/inference_session.py"
_PERFMODEL = _SRC / "aisimulate/sweeper/afd_perfmodel.py"


def test_afd_result_dict_publishes_layer_measurements():
    # Source-text coupling: importing inference_session pulls in the native runtime, which this
    # file deliberately avoids so the canary stays runnable in a bare checkout.
    assert '"afd_layer_measurements": afd_layer_measurements,' in _SESSION.read_text()
    assert "columns=common.ColumnsAFD" in _SESSION.read_text()


def test_columns_afd_drops_layer_measurements():
    """Canary: fails once the schema preserves the key, meaning the defect is fixed."""
    assert "afd_layer_measurements" not in common.ColumnsAFD


def test_sweeper_still_treats_the_key_as_required():
    """Blast radius: the sweeper rejects an estimate whose raw lost the measurements."""
    source = _PERFMODEL.read_text()
    assert 'raw.get("afd_layer_measurements")' in source
    assert "AIC estimate did not return afd_layer_measurements" in source
