#!/bin/bash
# Run tests in separate processes to avoid global state contamination in CocoIndex
set -e
echo "Running MCP tools test..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_mcp_tools
echo "Running pipeline and search test..."
uv run python -m unittest tests.test_pipeline.TestPocketPipeline.test_pipeline_and_search
echo "All tests passed successfully!"
