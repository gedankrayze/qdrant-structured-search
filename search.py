#!/usr/bin/env python3
"""
CLI tool for searching JSON documents using natural language queries.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_result(doc: Dict[str, Any], score: float) -> str:
    """Format a single search result for display.

    Args:
        doc: Document data
        score: Match score

    Returns:
        Formatted string representation
    """
    try:
        # Extract and format document sections
        sections = []

        def format_dict(d: Dict[str, Any], prefix: str = "") -> List[str]:
            """Recursively format dictionary key-value pairs."""
            lines = []
            for key, value in d.items():
                if isinstance(value, dict):
                    subsections = format_dict(value, f"{prefix}{key}.")
                    lines.extend(subsections)
                elif isinstance(value, list):
                    if value and isinstance(value[0], dict):
                        # Handle list of objects
                        for i, item in enumerate(value, 1):
                            subsections = format_dict(item, f"{prefix}{key}[{i}].")
                            lines.extend(subsections)
                    else:
                        # Handle simple lists
                        formatted_value = ", ".join(str(v) for v in value)
                        lines.append(f"{prefix}{key}: {formatted_value}")
                else:
                    lines.append(f"{prefix}{key}: {value}")
            return lines

        # Format all document fields
        sections.extend(format_dict(doc))

        # Create the formatted output
        formatted_result = "\n".join(sections)

        return f"""
[Match Score: {score:.3f}]
================================================================================
{formatted_result}
================================================================================
"""
    except Exception as e:
        logger.warning(f"Error formatting result: {e}, falling back to JSON")
        return f"""
[Match Score: {score:.3f}]
================================================================================
{json.dumps(doc, indent=2)}
================================================================================
"""


def search_documents(
        query: str,
        schema_path: Optional[str] = None,
        limit: int = 5,
        min_score: float = 0.001
) -> List[Tuple[float, Dict[str, Any]]]:
    """Search documents using the query and return results.

    Args:
        query: Search query string
        schema_path: Optional path to JSON schema for structured search
        limit: Maximum number of results
        min_score: Minimum similarity score threshold

    Returns:
        List of (score, document) tuples
    """
    try:
        # Initialize vector store with optional schema
        vector_store = VectorStore(schema_path=schema_path)
        logger.info(f"Searching with query: '{query}' (min_score={min_score})")

        results = vector_store.search(
            query=query,
            k=limit,
            score_threshold=min_score
        )

        if not results:
            logger.warning("No results found above minimum score threshold")
        else:
            logger.info(f"Found {len(results)} matching documents")
            for score, doc in results:
                title = doc.get('title', 'Untitled')
                logger.debug(f"Match (score={score:.3f}): {title}")

        return results

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Search JSON documents using natural language",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "query",
        help="Natural language search query"
    )

    parser.add_argument(
        "--schema",
        help="Path to JSON schema file for structured search"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to return"
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.001,
        help="Minimum similarity score threshold (0.0 to 1.0)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)

    try:
        # Validate schema path if provided
        schema_path = None
        if args.schema:
            schema_path = Path(args.schema)
            if not schema_path.exists():
                raise FileNotFoundError(f"Schema file not found: {args.schema}")

        # Perform search
        results = search_documents(
            query=args.query,
            schema_path=schema_path,
            limit=args.limit,
            min_score=args.min_score
        )

        # Handle no results
        if not results:
            print("No matching documents found.")
            sys.exit(0)

        # Output results
        if args.json:
            # JSON output format
            json_results = [
                {
                    "score": score,
                    "document": doc
                }
                for score, doc in results
            ]
            print(json.dumps(json_results, indent=2))
        else:
            # Human-readable output format
            print(f"\nFound {len(results)} matching documents:\n")
            for score, doc in results:
                print(format_result(doc, score))

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
