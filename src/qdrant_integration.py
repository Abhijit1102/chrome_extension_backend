from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import os
import asyncio
from dotenv import load_dotenv
from src.web_loader import load_web_content, adjust_chunk_size_for_embedding
from src.embeddings import HuggingFaceEmbedding

load_dotenv()

class QdrantManager:
    """
    A class to manage Qdrant operations such as collection management,
    embedding uploads, querying, and collection deletion.
    """

    def __init__(self, url=None, api_key=None):
        """
        Initialize the Qdrant client with API URL and key.

        Args:
            url (str): Qdrant API URL.
            api_key (str): Qdrant API Key.
        """
        self.qdrant_url = url or os.getenv("QDRANT_URL")
        self.api_key = api_key or os.getenv("QDRANT_API_KEY")
        self.collection = "web_embeddings"  
        self.vector_size = 384

        if not self.qdrant_url or not self.api_key:
            raise ValueError("⚠️ Qdrant URL or API key is missing. Check your .env file!")

        # Initialize Qdrant client
        self.client = QdrantClient(url=self.qdrant_url, api_key=self.api_key)

    def create_collection(self):
        """
        Create a new Qdrant collection if it doesn't exist.
        """
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={"size": self.vector_size, "distance": "Cosine"},
            )
            print(f"✅ Collection '{self.collection}' created successfully!")
        else:
            print(f"🔁 Collection '{self.collection}' already exists.")

    def upload_embeddings(self, embeddings, texts):
        """
        Upload embeddings to Qdrant collection.

        Args:
            embeddings (list): List of embeddings (vectors).
            texts (list): List of corresponding texts.
        """
        if len(embeddings) != len(texts):
            raise ValueError("❗ Embeddings and texts list must be of the same length!")

        points = [
            PointStruct(id=i, vector=embeddings[i], payload={"text": texts[i]})
            for i in range(len(embeddings))
        ]

        self.client.upsert(collection_name=self.collection, points=points)
        print(f"✅ Embeddings uploaded successfully to collection '{self.collection}'")

    def query_similar_texts(self, query_vector, limit=5):
        """
        Query the most similar texts from Qdrant.

        Args:
            query_vector (list): Vector to search for similar results.
            limit (int): Number of nearest neighbors to retrieve.

        Returns:
            list: List of most similar texts.
        """
        if not self.client.collection_exists(self.collection):
            raise ValueError(f"⚠️ Collection '{self.collection}' does not exist!")

        search_results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=limit,
            with_payload=True, 
        )

        if not search_results:
            print("❌ No results found!")
            return []

        # Extract and return the most similar texts
        similar_texts = []
        for result in search_results:
            if "text" in result.payload:
                similar_texts.append(result.payload["text"])

        print(f"🔎 Found {len(similar_texts)} similar texts.")
        return similar_texts

    def delete_collection(self):
        """
        Delete the collection from Qdrant if it exists.
        """
        try:
            if self.client.collection_exists(self.collection):
                self.client.delete_collection(collection_name=self.collection)
                print(f"❌ Collection '{self.collection}' deleted successfully!")
            else:
                print(f"⚠️ Collection '{self.collection}' does not exist.")
        except Exception as e:
            print(f"⚠️ Error: {e}")

    async def process_and_upload_chunks(self, web_url, embedding_model):
        """
        Process web content, generate embeddings, and upload them to Qdrant.

        Args:
            web_url (str): URL to load content from.
            embedding_model (HuggingFaceEmbedding): Embedding model to generate embeddings.

        Returns:
            None
        """
        try:
            # Load content from the web
            docs = load_web_content(web_url)

            if docs:
                all_chunks = []
                all_embeddings = []

                for doc in docs:
                    text_content = doc.page_content
                    chunks = adjust_chunk_size_for_embedding(text_content)

                    embeddings = await embedding_model.get_embeddings(chunks)

                    all_chunks.extend(chunks)
                    all_embeddings.extend(embeddings)

                # Upload embeddings to Qdrant
                if all_chunks and all_embeddings:
                    self.upload_embeddings(all_embeddings, all_chunks)
                    print(f"✅ Successfully uploaded {len(all_chunks)} chunks to '{self.collection}' collection!")

        except Exception as e:
            print(f"❌ Error while processing and uploading chunks: {e}")
