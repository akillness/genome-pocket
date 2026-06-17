#!/bin/bash
# Run tests in separate processes to avoid global state contamination in CocoIndex
set -e
echo "Running MCP tools test..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_mcp_tools
echo "Running pipeline and search test..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_pipeline_and_search
echo "Running incremental memoization test (DoD #3)..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_incremental_memoization
echo "Running deletion propagation test (DoD #4)..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_deletion_propagates
echo "Running transaction rollback test (abort_source)..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_abort_source_discards_uncommitted_rows
echo "All tests passed successfully!"
