#!/usr/bin/env python3
"""
Run only the performance benchmark against existing indexed content.
"""

import asyncio
import sys
from pathlib import Path

# Add semantic search module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from semantic_search_benchmark import SemanticSearchBenchmark, print_benchmark_results

# Database connection details
DATABASE_URL = "postgresql://kcs:kcs_dev_password_change_in_production@localhost:5432/kcs"

async def main():
    """Main function to run benchmark."""
    print("🏃‍♂️ Running performance benchmark against real semantic search data...")
    print("="*80)

    # Run the performance benchmark
    benchmark = SemanticSearchBenchmark(DATABASE_URL)
    try:
        results = await benchmark.run_full_benchmark()
        print_benchmark_results(results)

        # Exit with appropriate code
        if results.get("overall_assessment", {}).get("constitutional_compliance", False):
            print("\n🎉 Performance benchmark PASSED!")
            return 0
        else:
            print("\n❌ Performance benchmark FAILED!")
            return 1

    except Exception as e:
        print(f"\n💥 Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
