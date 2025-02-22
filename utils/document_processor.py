import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from textwrap import dedent
from typing import Dict, Any, List, Optional, Union, Set

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logger = logging.getLogger(__name__)


class DocumentProcessor:
    def __init__(self, schema_path: Optional[str] = None, max_documents: int = 10000):
        """Initialize DocumentProcessor with optional schema and configuration."""
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.max_documents = max_documents
        self.document_timestamps: Dict[str, datetime] = {}

        # Store API key
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Load schema if provided
        self.schema = None
        if schema_path:
            try:
                schema_path = Path(schema_path)
                if not schema_path.exists():
                    raise FileNotFoundError(f"Schema file not found: {schema_path}")

                with schema_path.open('r') as f:
                    self.schema = json.load(f)
                logger.info(f"Successfully loaded JSON schema from {schema_path}")
            except Exception as e:
                logger.error(f"Failed to load schema: {e}")
                raise

    def _generate_document_id(self, document: Dict[str, Any], prefix: str = "") -> str:
        """Generate a unique document ID with optional prefix.

        Args:
            document: Document to generate ID for
            prefix: Optional prefix for the ID

        Returns:
            Unique document ID
        """
        # Use relevant fields for ID generation if schema is available
        if self.schema and "idFields" in self.schema:
            id_parts = []
            for field in self.schema["idFields"]:
                value = document.get(field, "")
                id_parts.append(str(value))
            id_source = "_".join(id_parts)
        else:
            # Fallback to content-based hash
            id_source = json.dumps(document, sort_keys=True)

        # Generate hash and combine with prefix and timestamp
        doc_hash = hashlib.blake2b(id_source.encode(), digest_size=16).hexdigest()
        timestamp = datetime.now().strftime("%Y%m%d")
        return f"{prefix}{timestamp}_{doc_hash}"

    def _cleanup_old_documents(self):
        """Remove oldest documents when limit is reached."""
        if len(self.documents) > self.max_documents:
            # Sort by timestamp and remove oldest
            sorted_docs = sorted(
                self.document_timestamps.items(),
                key=lambda x: x[1]
            )
            docs_to_remove = len(self.documents) - self.max_documents

            for doc_id, _ in sorted_docs[:docs_to_remove]:
                del self.documents[doc_id]
                del self.document_timestamps[doc_id]

            logger.info(f"Cleaned up {docs_to_remove} old documents")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _generate_text_with_llm(self, document: Dict[str, Any]) -> str:
        """Generate text representation of document using LLM with retry logic."""
        try:
            # Prepare context-aware prompt
            prompt = dedent(f"""
            Convert this JSON document into well-structured plain text. Focus on:
            1. Natural, readable format
            2. Preserving all important information
            3. Maintaining relationships between data
            4. Using appropriate sections and structure

            {f'Schema:\n{json.dumps(self.schema, indent=2)}\n' if self.schema else ''}

            Document:
            {json.dumps(document, indent=2)}
            """)

            # Calculate expected token count
            estimated_tokens = len(prompt.split()) * 1.5

            # Choose appropriate model based on size
            model = "gpt-4" if estimated_tokens > 4000 else "gpt-3.5-turbo"

            # Create async client
            async with AsyncOpenAI(api_key=self.api_key) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{
                        "role": "system",
                        "content": "You are a precise document formatter that converts JSON to readable text."
                    }, {
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.3,
                    max_tokens=4000
                )

                text = response.choices[0].message.content.strip()
                logger.debug(f"Generated text sample: {text[:200]}...")
                return text

        except Exception as e:
            logger.error(f"Error in LLM text generation: {e}")
            raise

    def process_document(self, document: Dict[str, Any], prefix: str = "") -> str:
        """Process a document and return its unique identifier.

        Args:
            document: Document to process
            prefix: Optional prefix for document ID

        Returns:
            Unique document ID
        """
        # Validate against schema if available
        if self.schema:
            pass
            # validate(instance=document, schema=self.schema)

        # Generate ID and store document
        doc_id = self._generate_document_id(document, prefix)
        self.documents[doc_id] = document
        self.document_timestamps[doc_id] = datetime.now()

        # Cleanup if needed
        self._cleanup_old_documents()

        return doc_id

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a document by its ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document if found, None otherwise
        """
        return self.documents.get(doc_id)

    def get_field_values(
            self,
            field_path: str,
            return_type: type = str
    ) -> Set[Union[str, int, float]]:
        """Get unique values for a specific field across all documents.

        Args:
            field_path: Dot-notation path to the field
            return_type: Expected type of the values

        Returns:
            Set of unique values
        """
        values = set()
        keys = field_path.split('.')

        def extract_value(obj: Any, remaining_keys: List[str]) -> None:
            if not remaining_keys:
                if obj is not None:
                    try:
                        values.add(return_type(obj))
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert value {obj} to {return_type}")
                return

            key = remaining_keys[0]
            if isinstance(obj, dict):
                if key in obj:
                    extract_value(obj[key], remaining_keys[1:])
            elif isinstance(obj, list):
                for item in obj:
                    extract_value(item, remaining_keys)

        for doc in self.documents.values():
            extract_value(doc, keys)

        return values

    def __len__(self) -> int:
        """Return the number of documents currently stored."""
        return len(self.documents)

    def __contains__(self, doc_id: str) -> bool:
        """Check if a document ID exists in the store."""
        return doc_id in self.documents
