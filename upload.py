#!/usr/bin/env python3
"""
CLI tool for uploading JSON documents to the vector store.
"""
import argparse
import asyncio
import json
import logging
import sys
from os.path import basename
from pathlib import Path
from typing import List, Dict, Optional

from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> dict:
    """Load and validate a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Parsed JSON document

    Raises:
        JSONDecodeError: If file contains invalid JSON
        FileNotFoundError: If file doesn't exist
    """
    try:
        with file_path.open('r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise


async def process_file(
        file_path: Path,
        doc_processor: DocumentProcessor,
        vector_store: VectorStore,
        prefix: str = ""
) -> Dict[str, str]:
    """Process a single JSON file.

    Args:
        file_path: Path to JSON file
        doc_processor: Document processor instance
        vector_store: Vector store instance
        prefix: Optional prefix for document IDs

    Returns:
        Dictionary with processing status and details
    """
    result = {
        "path": str(file_path),
        "status": "failed",
        "error": None,
        "id": None
    }

    try:
        logger.info(f"Processing {file_path}")

        # Load and validate JSON
        json_doc = load_json_file(file_path)
        json_doc["_source"] = basename(file_path)

        # Process document
        doc_id = doc_processor.process_document(json_doc, prefix=prefix)

        # Generate text representation using LLM
        doc_text = await doc_processor._generate_text_with_llm(json_doc)
        if not doc_text:
            raise ValueError("Failed to generate document text")

        # Add to vector store
        vector_store.add_document(doc_id, doc_text, json_doc)

        result.update({
            "status": "success",
            "id": doc_id,
            "error": None
        })
        logger.info(f"Successfully processed {file_path} (ID: {doc_id})")

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {str(e)}")
        result["error"] = str(e)

    return result


async def process_files(
        file_paths: List[Path],
        schema_path: Optional[str] = None,
        prefix: str = "",
        batch_size: int = 10
) -> Dict:
    """Process multiple JSON files with batching support.

    Args:
        file_paths: List of paths to JSON files
        schema_path: Optional path to JSON schema
        prefix: Optional prefix for document IDs
        batch_size: Number of files to process concurrently

    Returns:
        Processing statistics
    """
    stats = {
        "processed": 0,
        "failed": 0,
        "files": []
    }

    try:
        # Initialize processors with schema if provided
        doc_processor = DocumentProcessor(schema_path=schema_path)
        vector_store = VectorStore(schema_path=schema_path)

        # Process files in batches
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]

            # Process batch concurrently
            tasks = [
                process_file(
                    file_path,
                    doc_processor,
                    vector_store,
                    prefix
                )
                for file_path in batch
            ]

            results = await asyncio.gather(*tasks)

            # Update statistics
            for result in results:
                stats["files"].append(result)
                if result["status"] == "success":
                    stats["processed"] += 1
                else:
                    stats["failed"] += 1

        return stats

    except Exception as e:
        logger.error(f"Error during batch processing: {str(e)}")
        raise


def validate_paths(paths: List[str]) -> List[Path]:
    """Validate and expand file paths.

    Args:
        paths: List of path strings (may include wildcards)

    Returns:
        List of validated Path objects

    Raises:
        ValueError: If no valid files found
    """
    valid_paths = []

    for path_str in paths:
        path = Path(path_str)
        if path.is_file():
            valid_paths.append(path)
        else:
            # Handle wildcards
            matches = list(Path().glob(path_str))
            if not matches:
                logger.warning(f"No files found matching: {path_str}")
            valid_paths.extend([p for p in matches if p.is_file()])

    if not valid_paths:
        raise ValueError("No valid files to process")

    return valid_paths


async def main():
    parser = argparse.ArgumentParser(
        description="Upload JSON documents to vector store",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "files",
        nargs="+",
        help="JSON files to process (accepts multiple files or wildcards)"
    )

    parser.add_argument(
        "--schema",
        help="Path to JSON schema file"
    )

    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix for document IDs"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of files to process concurrently"
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
        # Validate and expand file paths
        file_paths = validate_paths(args.files)
        logger.info(f"Found {len(file_paths)} files to process")

        # Process files
        stats = await process_files(
            file_paths,
            schema_path=args.schema,
            prefix=args.prefix,
            batch_size=args.batch_size
        )

        # Print summary
        print("\nProcessing Summary:")
        print(f"Successfully processed: {stats['processed']}")
        print(f"Failed: {stats['failed']}")

        if stats["failed"] > 0:
            print("\nFailed files:")
            for file in stats["files"]:
                if file["status"] == "failed":
                    print(f"- {file['path']}: {file['error']}")

        sys.exit(0 if stats["failed"] == 0 else 1)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
