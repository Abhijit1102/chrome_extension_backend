from fastapi import FastAPI, HTTPException, BackgroundTasks
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


async def process_url_background(web_url: str):
    try:
        qdrant_manager.create_collection()
        await qdrant_manager.process_and_upload_chunks(web_url, embedding)
        print("✅ URL content processed and stored successfully.")
        
        await send_processing_status("Tab content processed successfully.")
    except Exception as e:
        print(f"❌ Error processing URL: {str(e)}")
        await send_processing_status(f"Error processing URL: {str(e)}")

async def send_processing_status(message: str):
    print(f"📩 Status Update: {message}")


@app.post("/process_url")
async def process_url(request: CollectionRequest, background_tasks: BackgroundTasks):
    """Process content from active tab URL asynchronously."""
    global web_url 
    try:
        web_url = request.url  
        background_tasks.add_task(process_url_background, web_url)

        return {"status": "Processing started. Please wait for completion."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting processing: {str(e)}")


@app.post("/get_answer")
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


@app.post("/delete_collection")
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
