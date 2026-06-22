#!/bin/bash
# Run the full genome-pocket test suite.
#
# The suite is designed around the session-scoped, autouse MockEmbedder fixture
# in tests/conftest.py, which swaps every embedding path to a deterministic,
# offline fake. That fixture is a *pytest* fixture, so the suite MUST be driven
# by pytest — running individual classes via `python -m unittest` silently
# bypasses the patch and falls back to downloading/loading real model weights
# (slow, network-dependent, and non-deterministic).
#
# pytest auto-discovers every test module (test_pipeline, test_retrieval_api,
# test_graph_unit, test_multimodal), so this runner can never go stale as test
# classes are added or moved.
set -e

echo "Running full test suite (pytest, offline MockEmbedder)..."
uv run python -m pytest "$@"
echo "All tests passed successfully!"
