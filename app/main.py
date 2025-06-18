# /app/main.py

import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# --- Service Import ---
try:
    # This single import points to the main orchestrator in the service file
    from .api.openai_service import process_user_query
except ImportError:
    traceback.print_exc()
    raise RuntimeError("Could not import the AI service. Ensure the file structure is correct: /app/api/openai_service.py")

# --- App Setup ---
app = FastAPI(
    title="Sports Chatbot Microservice (SPAI) - v6.0 Two-Call",
    description="Provides sports info using a two-call (Gather -> Present) OpenAI architecture."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Models ---
class ChatRequest(BaseModel):
    user_id: str = "default_user"
    query: str

class ChatResponse(BaseModel):
    reply: str
    ui_data: Dict[str, Any]

# --- In-Memory History ---
conversation_histories: Dict[str, List[Dict[str, str]]] = {}
HISTORY_LIMIT = 10 # Store the last 5 user/assistant turns

# --- Routes ---
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def read_index():
    html_file_path = "app/static/htmlsim.html"
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="Index HTML not found.")
    return FileResponse(html_file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok_v6.0"}

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    print(f"--- /chat CALLED by user: {request.user_id}, query: '{request.query[:60]}...' ---")
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    user_id = request.user_id
    user_query = request.query
    current_history = conversation_histories.get(user_id, [])

    try:
        # --- Single entry point for the entire process ---
        final_result = await process_user_query(user_query, current_history)

        # --- Update History ---
        current_history.append({"role": "user", "content": user_query})
        if final_result and final_result.get("reply"):
            current_history.append({"role": "assistant", "content": final_result["reply"]})
        
        conversation_histories[user_id] = current_history[-HISTORY_LIMIT:]
        print(f">>> History for {user_id} updated. New length: {len(conversation_histories[user_id])}")

        # --- Return Response ---
        return ChatResponse(reply=final_result["reply"], ui_data=final_result["ui_data"])

    except Exception as e:
        error_type = type(e).__name__
        print(f"!!! UNHANDLED EXCEPTION in /chat endpoint: {error_type} - {e}")
        traceback.print_exc()
        return ChatResponse(
            reply=f"A critical server error ({error_type}) occurred. Please try again later.",
            ui_data={"component_type": "generic_text", "data": {"error": f"Server error: {error_type}"}}
        )

print("--- app/main.py: Application startup complete. ---")