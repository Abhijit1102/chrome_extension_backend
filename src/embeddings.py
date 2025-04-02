import aiohttp
import asyncio
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


class HuggingFaceEmbedding:
    """
    A class to interact with Hugging Face API to generate text embeddings asynchronously.
    """

    def __init__(self, model_id="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the HuggingFaceEmbedding class with API credentials and model ID.

        Args:
            model_id (str): ID of the embedding model. Default is 'all-MiniLM-L6-v2'.
        """
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("⚠️ Hugging Face API key is missing. Check your .env file!")

        self.model_id = model_id
        self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.model_id}"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}

    async def get_embeddings(self, texts):
        """
        Generate embeddings for a list of texts asynchronously using Hugging Face API.

        Args:
            texts (list): List of strings to embed.

        Returns:
            list: Embedding vectors for input texts.
        """
        # Check if input is a list
        if not isinstance(texts, list):
            raise TypeError("Input texts must be a list of strings.")

        # Prepare request payload
        payload = {"inputs": texts, "options": {"wait_for_model": True}}

        async with aiohttp.ClientSession() as session:
            async with session.post(self.api_url, headers=self.headers, json=payload) as response:
                if response.status != 200:
                    error_details = await response.json()
                    raise Exception(f"❌ Error: {response.status}, {error_details}")

                # Return embeddings asynchronously
                return await response.json()


