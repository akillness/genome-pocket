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
echo "Running run-stats / monitoring test..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_run_reports_stats
echo "Running live-mode watch test..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_live_mode_picks_up_new_file
echo "Running hybrid retrieval + REST API tests..."
uv run python -m unittest tests.test_retrieval_api.TestRetrievalAndApi
echo "Running text refiner unit tests..."
uv run python -m unittest tests.test_retrieval_api.TestTextRefiner
echo "Running code-aware splitting tests (POCKET-403)..."
uv run python -m unittest tests.test_retrieval_api.TestCodeAwareSplitting
echo "Running lifecycle command tests (POCKET-405)..."
uv run python -m unittest tests.test_retrieval_api.TestLifecycleCommands
echo "Running graph extraction / resolution ops tests (POCKET-404)..."
uv run python -m unittest tests.test_retrieval_api.TestGraphExtraction
echo "Running graph target tests (POCKET-404a)..."
uv run python -m unittest tests.test_retrieval_api.TestGraphTarget
echo "All tests passed successfully!"
