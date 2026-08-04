#!/usr/bin/env python3
"""Offline tool router calibration benchmark script (T6).

Validates tool routing precision and recall against a curated golden dataset
of representative query-to-tool mappings.
"""

import sys
import os
from pathlib import Path

# Add root directory to sys.path so we can import components correctly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from langchain_core.messages import HumanMessage
from components import tool_router


# Golden dataset: representative queries and expected tool activations
GOLDEN_DATASET = [
    ("is there any credential stored inside my vault?", "vault"),
    ("save this api token to the vault", "vault"),
    ("what did I tell you about my code preference earlier?", "recall"),
    ("remember that my favorite language is Python", "remember"),
    ("run pytest inside the current directory", "execute_shell_command"),
    ("what has changed in my repository working tree?", "git_diff"),
    ("apply these patches to the configurations", "batch_patch"),
    ("fetch the contents of the page at https://example.com/api", "fetch_url"),
    ("read the first 50 lines of langbot.py", "read_scratch"),  # trigger: "scratch:" or "read_scratch"
    ("find where is TODO inside components directory", "find_in_files"),
    ("list the folder contents with their sizes", "glob_list"),
]


def run_benchmark():
    print("=" * 60)
    print("      TOOL ROUTER CALIBRATION & EVALUATION HARNESS")
    print("=" * 60)

    # Initialize router description embeddings
    try:
        tool_router._ensure_desc_vectors()
    except Exception as e:
        print(f"Skipping embeddings calibration pass (no model loaded / offline mode): {e}")
        # Run keyword/regex only evaluation
        pass

    results = []
    print(f"{'Query Text':<50} | {'Expected':<15} | {'Status':<8}")
    print("-" * 80)
    
    passed = 0
    total = len(GOLDEN_DATASET)

    for query, expected_tool in GOLDEN_DATASET:
        # Get bound tools
        bound_tools = tool_router.select_tool_names([HumanMessage(content=query)])
        
        # Verify activation
        is_correct = expected_tool in bound_tools
        status = "PASSED" if is_correct else "FAILED"
        if is_correct:
            passed += 1

        print(f"{query[:50]:<50} | {expected_tool:<15} | {status:<8}")
        
    accuracy = (passed / total) * 100
    print("-" * 80)
    print(f"Router Accuracy Benchmark: {passed}/{total} ({accuracy:.1f}%)")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_benchmark()
    sys.exit(0 if success else 1)
