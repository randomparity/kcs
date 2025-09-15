#!/bin/bash
set -euo pipefail

# KCS Parser Benchmark Runner
# Runs comprehensive benchmarks and generates HTML reports

echo "🔥 Running KCS Parser Benchmarks"
echo "================================="

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Check if criterion is available
if ! cargo bench --help | grep -q "criterion"; then
    echo "Installing criterion for benchmarking..."
    cargo add --dev criterion --features html_reports
fi

# Create benchmark output directory
mkdir -p target/criterion

echo "📊 Running parser benchmarks..."
echo "This may take several minutes..."

# Run benchmarks with HTML reports
cargo bench --bench parser_benchmarks

echo ""
echo "✅ Benchmarks completed!"
echo ""

# Check if reports were generated
if [ -d "target/criterion" ]; then
    echo "📈 Benchmark reports generated:"
    find target/criterion -name "index.html" | head -5 | while read -r report; do
        echo "  - file://$(realpath "$report")"
    done

    echo ""
    echo "🌐 Main report: file://$(realpath target/criterion/reports/index.html)"

    # Generate summary statistics
    echo ""
    echo "📋 Quick Summary:"
    echo "=================="

    # Parse benchmark results for quick overview
    if [ -f "target/criterion/parser_benchmarks/single_file_parsing/tree_sitter_only/small/base/estimates.json" ]; then
        echo "✓ Single file parsing benchmarks completed"
    fi

    if [ -f "target/criterion/parser_benchmarks/batch_parsing/batch_content/1/base/estimates.json" ]; then
        echo "✓ Batch parsing benchmarks completed"
    fi

    if [ -f "target/criterion/parser_benchmarks/constitutional_requirements/large_file_parsing_target/base/estimates.json" ]; then
        echo "✓ Constitutional requirement benchmarks completed"
    fi

else
    echo "⚠️  No benchmark reports found in target/criterion"
fi

echo ""
echo "💡 Tips:"
echo "  - Open the HTML reports in your browser for detailed analysis"
echo "  - Re-run benchmarks after code changes to track performance"
echo "  - Use 'cargo bench -- --save-baseline <name>' to save baselines"
echo "  - Compare with 'cargo bench -- --baseline <name>' for regression testing"

echo ""
echo "🎯 Performance Targets (from constitution):"
echo "  - File parsing: <100ms for IDE integration"
echo "  - Large file parsing: <1000ms"
echo "  - Memory efficiency: Minimal allocations"
echo "  - Symbol accuracy: 100% extraction rate"

echo ""
echo "📝 Next Steps:"
echo "  1. Review HTML reports for performance bottlenecks"
echo "  2. Set up CI benchmarking with cargo bench"
echo "  3. Add more kernel-specific test cases"
echo "  4. Profile with 'cargo flamegraph' for detailed analysis"
