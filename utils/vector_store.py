import hashlib
import json
import logging
import os
from textwrap import dedent
from typing import List, Tuple, Dict, Any, Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Configure logging
logger = logging.getLogger(__name__)


class VectorStore:
    def __init__(self, schema_path: Optional[str] = None):
        """Initialize the vector store with OpenAI and Qdrant clients.

        Args:
            schema_path: Optional path to JSON schema file. If provided, enables schema-based filtering.
        """
        try:
            # Initialize OpenAI client with API key
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self.openai = OpenAI(api_key=api_key)

            # Load JSON schema if provided
            self.schema = None
            if schema_path:
                try:
                    with open(schema_path, 'r') as f:
                        self.schema = json.load(f)
                    logger.info(f"Successfully loaded JSON schema from {schema_path}")
                except Exception as e:
                    logger.error(f"Failed to load schema from {schema_path}: {e}")
                    raise ValueError(f"Invalid schema file: {e}")

            # Initialize Qdrant client
            qdrant_url = os.environ.get('QDRANT_URL')
            qdrant_api_key = os.environ.get('QDRANT_API_KEY')
            if not qdrant_url or not qdrant_api_key:
                raise ValueError("QDRANT_URL or QDRANT_API_KEY environment variables are not set")

            self.qdrant = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key
            )

            # Set collection parameters
            self.collection_name = os.environ.get('QDRANT_COLLECTION', "vector_store")
            self.dimension = 1536  # OpenAI ada-002 embedding dimension

            # Create collection only if it doesn't exist
            collections = self.qdrant.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if exists:
                logger.info(f"Using existing collection: {self.collection_name}")
            else:
                self._create_collection()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize VectorStore: {str(e)}")

    def _create_collection(self):
        """Create a new collection with necessary indexes."""
        try:
            # Create collection with vector configuration
            vectors_config = models.VectorParams(
                size=self.dimension,
                distance=models.Distance.COSINE
            )

            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=vectors_config,
                on_disk_payload=True
            )

            # Create payload index for document text search
            self.qdrant.create_payload_index(
                collection_name=self.collection_name,
                field_name="doc_text",
                field_schema=models.PayloadSchemaType.TEXT
            )

            # If schema is provided, create schema-specific indexes
            if self.schema:
                self._create_schema_specific_indexes()

            logger.info(f"Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def _create_schema_specific_indexes(self):
        """Create indexes based on the provided schema structure."""
        try:
            def create_indexes_recursive(schema: Dict[str, Any], path: List[str] = None):
                if path is None:
                    path = []

                properties = schema.get("properties", {})
                for field_name, field_schema in properties.items():
                    current_path = path + [field_name]
                    field_type = field_schema.get("type")

                    # Create appropriate index based on field type
                    if field_type == "object":
                        create_indexes_recursive(field_schema, current_path)
                    else:
                        field_path = "document." + ".".join(current_path)
                        try:
                            if field_type == "string":
                                self.qdrant.create_payload_index(
                                    collection_name=self.collection_name,
                                    field_name=field_path,
                                    field_schema=models.PayloadSchemaType.KEYWORD
                                )
                            elif field_type in ["integer", "number"]:
                                self.qdrant.create_payload_index(
                                    collection_name=self.collection_name,
                                    field_name=field_path,
                                    field_schema=models.PayloadSchemaType.INTEGER if field_type == "integer" else models.PayloadSchemaType.FLOAT
                                )
                            elif field_type == "array":
                                self.qdrant.create_payload_index(
                                    collection_name=self.collection_name,
                                    field_name=field_path,
                                    field_schema=models.PayloadSchemaType.KEYWORD
                                )
                        except Exception as e:
                            logger.warning(f"Failed to create index for {field_path}: {e}")

            create_indexes_recursive(self.schema)
            logger.info("Created schema-specific payload indexes")
        except Exception as e:
            logger.warning(f"Error creating schema-specific indexes: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embeddings using OpenAI API"""
        try:
            logger.debug(f"Generating embedding for text sample: {text[:200]}...")

            response = self.openai.embeddings.create(
                model="text-embedding-ada-002",
                input=text,
                encoding_format="float"
            )
            embedding = response.data[0].embedding

            logger.debug(f"Generated embedding with dimension {len(embedding)}")
            return embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def _generate_filter_conditions(self, query: str) -> Optional[models.Filter]:
        """Generate Qdrant filter conditions from natural language query using LLM if schema is available."""
        if not self.schema:
            logger.debug("No schema available, using semantic search only")
            return None

        try:
            logger.debug(f"Generating filters for query: {query}")

            # Prepare a prompt that includes the full schema context
            prompt = f"""As a search filter generator, analyze this query and create Qdrant-compatible filter conditions based on the provided JSON schema.
            The schema describes the structure of documents in our vector store.

            QUERY: {query}

            JSON SCHEMA:
            {json.dumps(self.schema, indent=2)}

            INSTRUCTIONS:
            1. Analyze the schema to understand the document structure
            2. Generate filter conditions that match Qdrant's filter syntax
            3. Only include filters for fields that are explicitly or implicitly mentioned in the query
            4. Use appropriate operators based on field types (range for numbers, match for keywords, etc.)
            5. Return null if no meaningful filters can be determined
            6. All document fields are prefixed with "document." in the filter conditions

            RESPONSE FORMAT:
            Return ONLY a JSON object with the following structure:
            {{
                "must": [
                    {{
                        "key": "document.path.to.field",
                        "range": {{ "gte": value, "lte": value }}  // For numeric fields
                    }},
                    {{
                        "key": "document.path.to.array",
                        "match": {{ "any": ["value1", "value2"] }}  // For array/enum fields
                    }},
                    {{
                        "key": "document.path.to.text",
                        "match": {{ "text": "value" }}  // For text fields
                    }}
                ]
            }}

            EXAMPLES:
            Query: "Show documents from 2023 about renewable energy"
            {{
                "must": [
                    {{
                        "key": "document.year",
                        "range": {{ "gte": 2023, "lte": 2023 }}
                    }},
                    {{
                        "key": "document.topics",
                        "match": {{ "any": ["renewable energy"] }}
                    }}
                ]
            }}

            Query: "Find high-priority tasks assigned to Alice"
            {{
                "must": [
                    {{
                        "key": "document.priority",
                        "match": {{ "any": ["high"] }}
                    }},
                    {{
                        "key": "document.assignee",
                        "match": {{ "text": "Alice" }}
                    }}
                ]
            }}

            Now, generate filter conditions for the provided query:"""

            # Get LLM response
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": dedent(
                            "You are a precise filter generator that creates Qdrant-compatible search filters based on JSON schemas. You only output valid JSON filter conditions or null.")},
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                response_format={"type": "json_object"}  # Ensure JSON response
            )

            # Extract and validate filter conditions
            filter_data = json.loads(response.choices[0].message.content)

            print(filter_data)

            if not filter_data or not isinstance(filter_data, dict) or "must" not in filter_data:
                logger.debug("No valid filter conditions generated")
                return None

            # Process filter conditions
            must_conditions = []
            for condition in filter_data.get("must", []):
                try:
                    if not isinstance(condition, dict) or "key" not in condition:
                        continue

                    field = condition["key"]

                    # Validate field against schema
                    field_path = field.replace("document.", "").split(".")
                    current_schema = self.schema
                    for part in field_path:
                        if "properties" not in current_schema or part not in current_schema["properties"]:
                            logger.warning(f"Invalid field path: {field}")
                            continue
                        current_schema = current_schema["properties"][part]

                    # Create appropriate filter based on schema type
                    if "range" in condition and current_schema.get("type") in ["number", "integer"]:
                        range_data = condition["range"]
                        range_params = {}

                        if isinstance(range_data, dict):
                            for op in ["gte", "lte", "gt", "lt"]:
                                if op in range_data and isinstance(range_data[op], (int, float)):
                                    range_params[op] = float(range_data[op])

                        if range_params:
                            must_conditions.append(
                                models.FieldCondition(
                                    key=field,
                                    range=models.Range(**range_params)
                                )
                            )
                            logger.debug(f"Added range condition: {field} = {range_params}")

                    elif "match" in condition:
                        match_data = condition["match"]
                        if isinstance(match_data, dict):
                            if "any" in match_data and isinstance(match_data["any"], list):
                                if current_schema.get("type") == "array" or "enum" in current_schema:
                                    must_conditions.append(
                                        models.FieldCondition(
                                            key=field,
                                            match=models.MatchAny(any=match_data["any"])
                                        )
                                    )
                                    logger.debug(f"Added match_any condition: {field} = {match_data['any']}")
                            elif "text" in match_data and isinstance(match_data["text"], str):
                                if current_schema.get("type") == "string":
                                    must_conditions.append(
                                        models.FieldCondition(
                                            key=field,
                                            match=models.MatchText(text=match_data["text"])
                                        )
                                    )
                                    logger.debug(f"Added match_text condition: {field} = {match_data['text']}")

                except Exception as e:
                    logger.warning(f"Error processing filter condition: {e}", exc_info=True)
                    continue

            if must_conditions:
                logger.info(f"Generated {len(must_conditions)} filter conditions")
                return models.Filter(must=must_conditions)

            logger.debug("No valid filter conditions generated")
            return None

        except Exception as e:
            logger.error(f"Error generating filter conditions: {e}", exc_info=True)
            return None

    def add_document(self, doc_id: str, text: str, document: Dict[str, Any]):
        """Add a document to the vector store.

        Args:
            doc_id: Unique identifier for the document
            text: Text content to be embedded
            document: Structured document data (must conform to schema if provided)
        """
        try:
            logger.info(f"Processing document {doc_id}")
            logger.debug(f"Document text sample: {text[:500]}...")

            # Generate embedding
            embedding = self._get_embedding(text)

            # Create payload
            payload = {
                "document": document,
                "doc_text": text
            }

            # Generate point ID from document ID
            point_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)

            # Add to Qdrant
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload=payload
                    )
                ]
            )
            logger.info(f"Successfully added document {doc_id} to vector store")
        except Exception as e:
            raise RuntimeError(f"Error adding document to vector store: {str(e)}")

    def search(self, query: str, k: int = 5, score_threshold: float = 0.001) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar documents using semantic search and optional filtering if schema is available.

        Args:
            query: Search query string
            k: Number of results to return
            score_threshold: Minimum similarity score threshold

        Returns:
            List of tuples containing (score, document) pairs
        """
        try:
            logger.info(f"Searching with query: {query}")
            query_vector = self._get_embedding(query)

            # Generate filter conditions if schema is available
            filter_conditions = None
            if self.schema:
                filter_conditions = self._generate_filter_conditions(query)
                if filter_conditions:
                    logger.debug(f"Using filter conditions: {filter_conditions}")
                else:
                    logger.debug("No filter conditions applied, using semantic search only")

            # Search in Qdrant
            # search_result = self.qdrant.search(
            #     collection_name=self.collection_name,
            #     query_vector=query_vector,
            #     limit=k * 2,  # Request extra results to account for filtering
            #     score_threshold=score_threshold,
            #     query_filter=filter_conditions
            # )
            search_result = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k * 3,  # Increase oversampling factor
                with_payload=True,
                with_vectors=False,
                query_filter=filter_conditions,
                score_threshold=score_threshold,
                search_params=models.SearchParams(
                    hnsw_ef=128  # Increase accuracy at slight performance cost
                )
            )

            # Process results
            results = []
            seen_docs = set()  # Track unique documents

            for hit in search_result:
                if 'document' in hit.payload:
                    doc = hit.payload['document']
                    doc_key = json.dumps(doc, sort_keys=True)  # Create unique key for document

                    if doc_key not in seen_docs:
                        seen_docs.add(doc_key)
                        results.append((hit.score, doc))

                        if len(results) >= k:
                            break

            logger.info(f"Found {len(results)} matching documents")
            return results[:k]

        except Exception as e:
            logger.error(f"Error performing search: {e}", exc_info=True)
            raise RuntimeError(f"Error performing search: {str(e)}")

    # Add query classification layer
    def _classify_query_type(self, query: str, use_llm: bool = False, classificator_llm: OpenAI = None) -> str:
        """Categorize query into types:
        - Fact lookup
        - Comparative analysis
        - Temporal analysis
        - Freeform exploration
        """
        logger.info(f"Classifying the query: {query}")

        # Use small LLM classifier or regex patterns
        if use_llm:
            # Use LLM to classify query
            if not classificator_llm:
                classificator_llm = self.openai
            pass
        pass

    # Add explainability to results
    def _generate_explanation(self, query: str, doc: dict, score: float, llm: OpenAI) -> str:
        """Generate natural language explanation of why document matched"""
        # Use LLM to compare query and document features
        if not llm:
            llm = self.openai
        pass

    def add_documents_batch(self, documents: List[Tuple[str, str, Dict[str, Any]]], batch_size: int = 100):
        """Add multiple documents to the vector store in batches.

        Args:
            documents: List of (doc_id, text, document) tuples
            batch_size: Number of documents to process in each batch
        """
        try:
            total_docs = len(documents)
            logger.info(f"Starting batch upload of {total_docs} documents")

            for i in range(0, total_docs, batch_size):
                batch = documents[i:i + batch_size]
                points = []

                for doc_id, text, document in batch:
                    try:
                        # Generate embedding
                        embedding = self._get_embedding(text)

                        # Generate point ID
                        point_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)

                        # Create point
                        points.append(
                            models.PointStruct(
                                id=point_id,
                                vector=embedding,
                                payload={
                                    "document": document,
                                    "doc_text": text
                                }
                            )
                        )

                    except Exception as e:
                        logger.error(f"Error processing document {doc_id}: {e}")
                        continue

                if points:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    logger.info(f"Successfully uploaded batch of {len(points)} documents")

            logger.info(f"Completed batch upload of documents")

        except Exception as e:
            raise RuntimeError(f"Error in batch document upload: {str(e)}")

    def delete_document(self, doc_id: str):
        """Delete a document from the vector store.

        Args:
            doc_id: Unique identifier of the document to delete
        """
        try:
            # Generate point ID from document ID
            point_id = int(hashlib.md5(doc_id.encode()).hexdigest()[:8], 16)

            # Delete from Qdrant
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[point_id]
                )
            )
            logger.info(f"Successfully deleted document {doc_id}")

        except Exception as e:
            raise RuntimeError(f"Error deleting document: {str(e)}")

    def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            self.qdrant.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter()
                )
            )
            logger.info(f"Successfully cleared collection {self.collection_name}")

        except Exception as e:
            raise RuntimeError(f"Error clearing collection: {str(e)}")

    def __enter__(self):
        """Context manager enter method."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        try:
            # Clean up OpenAI client if needed
            if hasattr(self, 'openai'):
                # Future cleanup if needed
                pass

            # Clean up Qdrant client
            if hasattr(self, 'qdrant'):
                self.qdrant.close()

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
