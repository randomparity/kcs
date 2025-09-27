#!/usr/bin/env python3
"""
Test suite for verifying JSONB metadata handling with automatic dict conversion.

This test verifies that the JSON codec configuration in the database connection
properly handles conversion between PostgreSQL JSONB and Python dicts.
"""

import asyncio
import json
import os
import sys
from typing import Any

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from src.python.semantic_search.database.connection import (
    DatabaseConfig,
    get_database_connection,
    init_database_connection,
)
from src.python.semantic_search.database.vector_store import (
    ContentFilter,
    SimilaritySearchFilter,
    VectorStore,
)


async def test_metadata_dict_handling():
    """Test that metadata is consistently handled as Python dicts."""
    print("=" * 80)
    print("METADATA DICT HANDLING TEST")
    print("=" * 80)

    # Initialize database connection
    config = DatabaseConfig.from_env()
    await init_database_connection(config)

    store = VectorStore()
    test_results = []

    try:
        # Test 1: Store content with complex nested metadata
        print("\nTest 1: Storing content with nested metadata...")
        complex_metadata = {
            "author": "John Doe",
            "tags": ["kernel", "memory", "optimization"],
            "version": "5.4.0",
            "nested": {
                "level1": {"level2": {"value": 42, "flag": True, "items": [1, 2, 3]}}
            },
            "null_value": None,
            "empty_list": [],
            "empty_dict": {},
        }

        content_id = await store.store_content(
            content_type="documentation",
            source_path=f"/test/metadata_test_{asyncio.get_event_loop().time()}.md",
            content="Test content for metadata handling verification",
            title="Metadata Test Document",
            metadata=complex_metadata,
        )

        print(f"✓ Stored content with ID: {content_id}")
        test_results.append(("Store complex metadata", True))

        # Test 2: Retrieve and verify metadata is a dict
        print("\nTest 2: Retrieving content and verifying dict type...")
        retrieved = await store.get_content_by_id(content_id)

        if retrieved:
            metadata = retrieved.metadata
            print(f"  Metadata type: {type(metadata)}")
            print(f"  Is dict: {isinstance(metadata, dict)}")

            assert isinstance(metadata, dict), f"Expected dict, got {type(metadata)}"
            assert metadata["author"] == "John Doe", "Author mismatch"
            assert metadata["tags"] == ["kernel", "memory", "optimization"], (
                "Tags mismatch"
            )
            assert metadata["nested"]["level1"]["level2"]["value"] == 42, (
                "Nested value mismatch"
            )
            assert metadata["null_value"] is None, "Null value not preserved"
            assert metadata["empty_list"] == [], "Empty list not preserved"
            assert metadata["empty_dict"] == {}, "Empty dict not preserved"

            print(
                "✓ Metadata correctly retrieved as Python dict with all values preserved"
            )
            test_results.append(("Retrieve as dict", True))
        else:
            print("✗ Failed to retrieve content")
            test_results.append(("Retrieve as dict", False))

        # Test 3: Test through similarity search
        print("\nTest 3: Testing metadata through similarity search...")

        # Create and store embedding
        dummy_embedding = [0.1] * 384
        await store.store_embedding(
            content_id=content_id,
            embedding=dummy_embedding,
            chunk_text="Test chunk for metadata",
            chunk_index=0,
            model_name="test",
            model_version="1.0",
        )

        # Search with filters
        filters = SimilaritySearchFilter(
            max_results=10, similarity_threshold=0.0, include_content=True
        )

        results = await store.similarity_search(dummy_embedding, filters)

        if results:
            result_metadata = results[0]["metadata"]
            print(f"  Search result metadata type: {type(result_metadata)}")

            assert isinstance(result_metadata, dict), (
                f"Expected dict in search, got {type(result_metadata)}"
            )
            assert result_metadata["author"] == "John Doe", (
                "Search metadata author mismatch"
            )

            # Test dict operations work directly
            author = result_metadata.get("author")
            tags_count = len(result_metadata.get("tags", []))
            nested_value = (
                result_metadata.get("nested", {})
                .get("level1", {})
                .get("level2", {})
                .get("value")
            )

            assert author == "John Doe", "Dict access failed"
            assert tags_count == 3, "Dict list access failed"
            assert nested_value == 42, "Nested dict access failed"

            print("✓ Similarity search returns proper Python dicts")
            test_results.append(("Similarity search dict", True))
        else:
            print("✗ No search results found")
            test_results.append(("Similarity search dict", False))

        # Test 4: Test empty/null metadata handling
        print("\nTest 4: Testing empty and null metadata...")

        # Store with null metadata
        null_content_id = await store.store_content(
            content_type="test",
            source_path=f"/test/null_metadata_{asyncio.get_event_loop().time()}.txt",
            content="Content with null metadata",
            metadata=None,
        )

        null_retrieved = await store.get_content_by_id(null_content_id)
        assert null_retrieved.metadata == {}, "Null metadata should become empty dict"
        print("✓ Null metadata handled correctly")

        # Store with empty dict
        empty_content_id = await store.store_content(
            content_type="test",
            source_path=f"/test/empty_metadata_{asyncio.get_event_loop().time()}.txt",
            content="Content with empty metadata",
            metadata={},
        )

        empty_retrieved = await store.get_content_by_id(empty_content_id)
        assert empty_retrieved.metadata == {}, "Empty metadata preserved"
        print("✓ Empty dict metadata handled correctly")
        test_results.append(("Empty/null metadata", True))

        # Test 5: Test metadata updates maintain dict type
        print("\nTest 5: Testing metadata modification...")

        # Modify the retrieved metadata (should be a dict we can change)
        if retrieved:
            retrieved.metadata["modified"] = True
            retrieved.metadata["tags"].append("updated")

            # This should work because it's a real Python dict
            assert retrieved.metadata["modified"] is True
            assert "updated" in retrieved.metadata["tags"]

            print("✓ Metadata dict is mutable and behaves like normal Python dict")
            test_results.append(("Dict mutability", True))

        # Test 6: Test raw database query returns dicts
        print("\nTest 6: Testing raw database queries...")

        db = get_database_connection()
        query = """
            SELECT id, metadata
            FROM indexed_content
            WHERE id = $1
        """

        async with db.acquire() as conn:
            row = await conn.fetchrow(query, content_id)
            if row:
                raw_metadata = row["metadata"]
                print(f"  Raw query metadata type: {type(raw_metadata)}")

                assert isinstance(raw_metadata, dict), (
                    f"Raw query should return dict, got {type(raw_metadata)}"
                )
                assert raw_metadata["author"] == "John Doe", (
                    "Raw query metadata mismatch"
                )

                print("✓ Raw database queries return Python dicts")
                test_results.append(("Raw query dict", True))

        # Cleanup
        print("\nCleaning up test data...")
        await store.delete_content(content_id)
        await store.delete_content(null_content_id)
        await store.delete_content(empty_content_id)
        print("✓ Test data cleaned up")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        test_results.append(("Exception", False))

    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(result[1] for result in test_results)

    if all_passed:
        print("\n🎉 All metadata handling tests passed!")
        print(
            "JSONB columns are properly converted to Python dicts throughout the application."
        )
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        exit(1)


async def test_metadata_type_verification():
    """Verify the specific type handling for different scenarios."""
    print("\n" + "=" * 80)
    print("METADATA TYPE VERIFICATION TEST")
    print("=" * 80)

    config = DatabaseConfig.from_env()
    await init_database_connection(config)

    store = VectorStore()

    # Test different data types in metadata
    test_cases = [
        ("String values", {"text": "hello world"}),
        ("Integer values", {"count": 42, "negative": -10}),
        ("Float values", {"pi": 3.14159, "scientific": 1.23e-4}),
        ("Boolean values", {"enabled": True, "disabled": False}),
        ("Null value", {"nothing": None}),
        ("Empty string", {"empty": ""}),
        (
            "Special characters",
            {"special": "Hello \"world\" with 'quotes' and \n newlines"},
        ),
        ("Unicode", {"unicode": "Hello 世界 🌍 مرحبا"}),
        ("Date-like string", {"date": "2024-01-01T12:00:00Z"}),
        (
            "Mixed nested",
            {
                "mixed": {
                    "int": 1,
                    "str": "two",
                    "list": [3, "four", 5.0],
                    "bool": True,
                    "null": None,
                }
            },
        ),
    ]

    print("\nTesting various data types in metadata:")
    for test_name, test_metadata in test_cases:
        try:
            # Store content with test metadata
            content_id = await store.store_content(
                content_type="test",
                source_path=f"/test/type_{test_name.replace(' ', '_')}_{asyncio.get_event_loop().time()}.txt",
                content=f"Testing {test_name}",
                metadata=test_metadata,
            )

            # Retrieve and verify
            retrieved = await store.get_content_by_id(content_id)

            if retrieved and retrieved.metadata == test_metadata:
                print(f"  ✅ {test_name}: Correctly preserved")
            else:
                print(f"  ❌ {test_name}: Mismatch!")
                print(f"     Expected: {test_metadata}")
                print(f"     Got: {retrieved.metadata if retrieved else 'None'}")

            # Cleanup
            await store.delete_content(content_id)

        except Exception as e:
            print(f"  ❌ {test_name}: Error - {e}")

    print("\n✓ Type verification complete")


async def main():
    """Run all metadata handling tests."""
    try:
        await test_metadata_dict_handling()
        await test_metadata_type_verification()

        print("\n" + "=" * 80)
        print("✅ ALL METADATA TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    # Configure logging for tests
    import logging

    logging.basicConfig(level=logging.INFO)

    # Run tests
    asyncio.run(main())
