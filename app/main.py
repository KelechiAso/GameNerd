# /app/main.py
'''
print("--- app/main.py: TOP OF FILE ---")
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, AsyncGenerator # Add AsyncGenerator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import traceback
import pprint

print("--- app/main.py: Standard imports DONE ---")

try:
    print("--- app/main.py: Attempting to import from api.openai_service ---")
    from .api.openai_service import (
        extract_sports_info,
        fetch_raw_text_data,
        get_structured_data_and_reply_via_tools
    )
    print("--- app/main.py: SUCCESSFULLY IMPORTED from api.openai_service ---")
except ImportError as e:
    print(f"--- app/main.py: !!! FAILED to import from api.openai_service due to ImportError: {e} !!! ---")
    traceback.print_exc() 
    raise e # CRITICAL: Re-raise to make app crash if service can't load
except Exception as e: 
    print(f"--- app/main.py: !!! FAILED to import from api.openai_service due to OTHER error: {e} !!! ---")
    traceback.print_exc()
    raise e # CRITICAL: Re-raise


app = FastAPI(
    title="Sports Chatbot Microservice (SPAI) - v2.1",
    description="Provides sports and gaming info using OpenAI tools and domain enforcement."
)
print("--- app/main.py: FastAPI INSTANCE CREATED ---")

# --- CORS ---
origins = ["*",]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("--- app/main.py: CORS Middleware ADDED ---")

# --- API Models ---
class ChatRequest(BaseModel):
    user_id: str = "default_user"
    query: str

class ChatResponse(BaseModel):
    reply: str
    ui_data: Dict[str, Any]
print("--- app/main.py: Pydantic Models DEFINED ---")

# --- In-Memory History Store ---
conversation_histories: Dict[str, List[Dict[str, str]]] = {}
HISTORY_LIMIT = 10
print("--- app/main.py: Conversation History Store INITIALIZED ---")

# --- Serve Static HTML File for the Root Path ---
@app.get("/", response_class=FileResponse)
async def read_index():
    print("--- app/main.py: / endpoint CALLED ---")
    html_file_path = "app/static/htmlsim.html"
    try:
        if not os.path.exists(html_file_path):
            print(f"--- app/main.py: ERROR - {html_file_path} NOT FOUND for / endpoint ---")
            raise HTTPException(status_code=404, detail="Index HTML not found. Ensure app/static/htmlsim.html exists.")
        print(f"--- app/main.py: Attempting to serve {html_file_path} for / endpoint ---")
        return FileResponse(html_file_path)
    except Exception as e:
        print(f"--- app/main.py: ERROR in / endpoint: {e} ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error serving index: {str(e)}")

@app.get("/health")
async def health_check():
    print("--- app/main.py: /health endpoint CALLED ---")
    return {"status": "ok_v2.1"}

# --- Main Chat Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    print(f"--- app/main.py: /chat endpoint CALLED by user: {request.user_id} with query: '{request.query[:30]}...' ---")
    user_id = request.user_id
    user_query = request.query
    if not user_query:
        print("--- app/main.py: /chat - Query is empty ---")
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Initialize defaults for this request
    final_reply = "Sorry, I'm having trouble processing sports/gaming requests right now."
    final_ui_data = {"component_type": "generic_text", "data": {"message": "Default error UI."}}
    current_history = conversation_histories.get(user_id, [])
    
    try:
        # --- Step 0: Extract Intent/Entities ---
        print(f"--- Step 0 (main.py): Extracting info for query: '{user_query[:50]}...' ---")
        parsed_info = await extract_sports_info(user_query, current_history)
        
        if not isinstance(parsed_info, dict) or parsed_info.get("input_type") == "error":
             error_msg = parsed_info.get('message', 'Query parsing failed at Step 0.')
             print(f"!!! Step 0 (main.py): Extraction failed or returned error: {error_msg}")
             return ChatResponse(reply=f"Error understanding your request: {error_msg}", ui_data=final_ui_data)
        
        print(f"--- Step 0 (main.py) Result (parsed_info):")
        pprint.pprint(parsed_info)

        input_type_from_step_0 = parsed_info.get("input_type")
        direct_reply_from_step_0 = parsed_info.get("conversation")
        short_circuit_types = ["acknowledgment", "simple_greeting", "invalid_request", "out_of_scope_request"]

        if input_type_from_step_0 in short_circuit_types and \
           direct_reply_from_step_0 and \
           isinstance(direct_reply_from_step_0, str) and \
           direct_reply_from_step_0.strip():
            print(f"--- Step 0 (main.py) identified as '{input_type_from_step_0}'. Short-circuiting with direct reply. ---")
            final_reply = direct_reply_from_step_0
            if input_type_from_step_0 == "out_of_scope_request":
                final_ui_data = {"component_type": "generic_text", "data": {"message": "This query is outside of my sports/gaming expertise."}}
            else: # For other short-circuits, keep ui_data generic
                final_ui_data = {"component_type": "generic_text", "data": {}}
        else:
            print(f"--- Step 1 (main.py): Attempting to fetch raw text data for input_type: {input_type_from_step_0} ---")
            raw_text_data = await fetch_raw_text_data(user_query, parsed_info, current_history)
            print(f"--- Step 1 (main.py) Result (raw_text_data snippet): {str(raw_text_data)[:150]}... ---")

            print(f"--- Step 2 (main.py): Generating reply and UI data via Tool Calling ---")
            final_data_dict = await get_structured_data_and_reply_via_tools(
                parsed_info,
                current_history,
                raw_text_data
            )

            if not isinstance(final_data_dict, dict) or "reply" not in final_data_dict or "ui_data" not in final_data_dict:
                 print(f"!!! Step 3 (main.py) ERROR: Tool calling step (Step 2) returned invalid dict structure. Response: {str(final_data_dict)[:200]}")
                 # final_reply uses its default error message
            else:
                 final_reply = final_data_dict.get("reply", "Processed your sports/gaming request.")
                 final_ui_data = final_data_dict.get("ui_data", {"component_type": "generic_text", "data": {"message": "No specific UI data was generated."}})
                 print(f">>> Step 3 (main.py): Successfully received reply/ui_data. UI Component: {final_ui_data.get('component_type')}")
        
        if not final_reply or not final_reply.strip(): # Ensure there's always some reply
            if input_type_from_step_0 == "acknowledgment": final_reply = "Noted."
            else: final_reply = "Your sports/gaming query has been processed."

        # --- Step 4: Update History ---
        print(f"--- Step 4 (main.py): Updating History ---")
        current_history.append({"role": "user", "content": user_query})
        if final_reply and final_reply.strip(): 
             current_history.append({"role": "assistant", "content": final_reply})
             print(f">>> Assistant reply added to history: '{final_reply[:100]}...'")
        else:
            print(f">>> Skipping assistant reply in history update due to empty content.")
        
        conversation_histories[user_id] = current_history[-HISTORY_LIMIT:]
        print(f">>> History for {user_id} updated. Length: {len(conversation_histories[user_id])}")

        final_response_object = ChatResponse(reply=final_reply, ui_data=final_ui_data)
        print(f"--- Request processing complete (main.py). Returning reply: '{final_reply[:50]}...' ---")
        return final_response_object

    except HTTPException as http_exc:
        print(f"!!! HTTP EXCEPTION CAUGHT directly in /chat: {http_exc.detail}")
        traceback.print_exc() # Log traceback for HTTPExceptions too
        raise http_exc 
    except Exception as e:
        error_type = type(e).__name__
        print(f"!!! UNHANDLED EXCEPTION in /chat endpoint (main.py) !!!")
        print(f"Exception Type: {error_type}")
        print(f"Exception details: {str(e)}")
        traceback.print_exc()
        # Ensure a fallback reply in case of unexpected errors
        error_reply = f"Sorry, a critical internal server error occurred ({error_type}). Please try a different sports/gaming query."
        error_ui_data = {"component_type": "generic_text", "data": {"error": f"Server error: {error_type}"}}
        return ChatResponse(reply=error_reply, ui_data=error_ui_data)

print("--- app/main.py: BOTTOM OF FILE, APP DEFINED ---")
'''

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
    from .api.openai_service import process_user_query
except ImportError:
    traceback.print_exc()
    raise RuntimeError("Could not import the AI service. Is the file structure correct?")

# --- App Setup ---
app = FastAPI(
    title="Sports Chatbot Microservice (SPAI) - v5.0 Single-Call",
    description="Provides sports info using a single-call, tool-based OpenAI architecture."
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
HISTORY_LIMIT = 10

# --- Routes ---
@app.get("/", response_class=FileResponse, include_in_schema=False)
async def read_index():
    html_file_path = "app/static/htmlsim.html"
    if not os.path.exists(html_file_path):
        raise HTTPException(status_code=404, detail="Index HTML not found.")
    return FileResponse(html_file_path)

@app.get("/health")
async def health_check():
    return {"status": "ok_v5.0"}

@app.post("/chat", response_model=ChatResponse)
async def handle_chat(request: ChatRequest):
    print(f"--- /chat CALLED by user: {request.user_id}, query: '{request.query[:50]}...' ---")
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    user_id = request.user_id
    user_query = request.query
    current_history = conversation_histories.get(user_id, [])

    try:
        # --- Single-step processing ---
        final_result = await process_user_query(user_query, current_history)

        # --- Update History ---
        current_history.append({"role": "user", "content": user_query})
        if final_result.get("reply"):
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
            reply=f"A critical server error ({error_type}) occurred. Please contact support.",
            ui_data={"component_type": "generic_text", "data": {"error": f"Server error: {error_type}"}}
        )

print("--- app/main.py: Application startup complete. ---")