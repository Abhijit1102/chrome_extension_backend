from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from datetime import datetime
import os


class MongoDB:
    """MongoDB connection and CRUD operations handler."""

    def __init__(self, db_name="chatbot", collection_name="chat_logs"):
        """Initialize MongoDB connection and select database & collection."""
        load_dotenv()
        mongo_uri = os.getenv("MONGODB_URL")

        if not mongo_uri:
            raise ValueError("⚠️ MONGODB_URL is not set in the environment variables.")
        
        self.client = AsyncIOMotorClient(mongo_uri)

        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    async def insert_log(self, url: str, user_query: str, bot_response: str):
        """Insert a chat log into MongoDB."""
        log = {
            "url": url,
            "user_query": user_query,
            "bot_response": bot_response,
            "timestamp": datetime.utcnow()
        }
        result = await self.collection.insert_one(log)
        return result.inserted_id