from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.embeddings import HuggingFaceEmbedding
from src.qdrant_integration import QdrantManager
from src.chatbot import QAChatBot
from src.database import MongoDB

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

mongo_db = MongoDB()
embedding = HuggingFaceEmbedding()
qdrant_manager = QdrantManager()
chatbot = QAChatBot()

qdrant_manager.create_collection()

web_url = ""

class CollectionRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query_text: str

@app.get("/api/v1/health")
def health_check():
    """Health check route to verify if the API is running."""
    return {"status": "Healthy", "message": "API is up and running."}

@app.post("/api/v1/process_url")
async def process_url(request: CollectionRequest):
    """Process content from the provided URL synchronously."""
    global web_url
    web_url = request.url  

    try:
        qdrant_manager.create_collection()
        await qdrant_manager.process_and_upload_chunks(web_url, embedding)
        print(f"✅ URL '{web_url}' content processed and stored successfully.")
        return {"status": "Success", "answer": "Your content is loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing URL '{web_url}': {str(e)}")

@app.post("/api/v1/get_answer")
async def get_answer(request: QueryRequest):
    """Get an AI-generated answer based on similar texts and user query."""
    global web_url
    try:
        query_embedding = (await embedding.get_embeddings([request.query_text]))[0]
        similar_texts = qdrant_manager.query_similar_texts(query_embedding, limit=5)
        context_data = "\n".join(similar_texts) if similar_texts else "No relevant context found."
        answer = chatbot.get_answer(request.query_text, context_data)
        await mongo_db.insert_log(web_url, request.query_text, answer)

        return {"status": "Success", "answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


@app.post("/api/v1/delete_collection")
async def delete_collection():
    """Delete the active Qdrant collection."""
    try:
        qdrant_manager.delete_collection()
        print("✅ Collection deleted successfully.")
        return {"status": "Success", "message": "Active collection deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
