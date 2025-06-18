# /app/api/openai_service.py

import json
import openai
import os
import traceback
import re
from dotenv import load_dotenv
from typing import Dict, List, Any

from openai import AsyncOpenAI, Timeout

# --- Setup ---
print("--- openai_service.py: TOP OF FILE (Two-Call Architecture) ---")
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("CRITICAL: OPENAI_API_KEY is not set in openai_service.py.")

client = AsyncOpenAI(
    api_key=openai_api_key,
    timeout=Timeout(180.0, connect=15.0) # Increased timeout for potentially longer calls
)
print("--- openai_service.py: AsyncOpenAI client INITIALIZED ---")


# --- Schemas & Tools (Re-used from original file) ---
# These definitions are crucial for the second call to structure data reliably.
SCHEMA_DATA_H2H = {
    "type": "object", "title": "H2HData", "description": "Data for head-to-head comparisons.",
    "properties": {
        "h2h_summary": {"type": "object", "properties": {
                "team1": {"type": "object", "properties": {"name": {"type": "string"}, "wins": {"type": ["integer", "null"]}, "draws": {"type": ["integer", "null"]}, "losses": {"type": ["integer", "null"]}, "goals_for": {"type": ["integer", "null"]}, "goals_against": {"type": ["integer", "null"]}}, "required": ["name"]},
                "team2": {"type": "object", "properties": {"name": {"type": "string"}, "wins": {"type": ["integer", "null"]}, "draws": {"type": ["integer", "null"]}, "losses": {"type": ["integer", "null"]}, "goals_for": {"type": ["integer", "null"]}, "goals_against": {"type": ["integer", "null"]}}, "required": ["name"]},
                "total_matches": {"type": ["integer", "null"]}}, "required": ["team1", "team2"]},
        "recent_meetings": {"type": "array", "items": {"type": "object", "properties": {"date": {"type": "string", "format": "date"}, "score": {"type": "string"}, "competition": {"type": "string"}}, "required": ["date", "score"]}}},
    "required": ["h2h_summary"]
}
SCHEMA_DATA_MATCH_SCHEDULE_TABLE = {
    "type": "object", "title": "MatchScheduleTableData", "description": "Data for a table of upcoming matches.",
    "properties": {"title": {"type": "string"}, "headers": {"type": "array", "items": {"type": "string"}},
                   "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                   "sort_info": {"type": ["string", "null"]}},
    "required": ["headers", "rows"]
}
SCHEMA_DATA_STANDINGS_TABLE = {
    "type": "object", "title": "StandingsTableData", "description": "Data for a league standings table.",
    "properties": {"league_name": {"type": "string"}, "season": {"type": ["string", "null"]},
                   "standings": {"type": "array", "items": {"type": "object", "properties": {
                       "rank": {"type": ["integer", "string"]}, "team_name": {"type": "string"}, "logo_url": {"type": ["string", "null"], "format": "uri"},
                       "played": {"type": "integer"}, "wins": {"type": "integer"}, "draws": {"type": "integer"}, "losses": {"type": "integer"},
                       "goals_for": {"type": "integer"}, "goals_against": {"type": "integer"}, "goal_difference": {"type": "integer"}, "points": {"type": "integer"},
                       "form": {"type": ["string", "null"]}}, "required": ["rank", "team_name", "played", "points"]}}},
    "required": ["league_name", "standings"]
}
SCHEMA_DATA_PLAYER_PROFILE = {
    "type": "object", "title": "PlayerProfileData", "description": "Detailed profile information for a specific player.",
    "properties": {"full_name": {"type": "string"}, "common_name": {"type": ["string", "null"]}, "nationality": {"type": "string"}, "date_of_birth": {"type": "string", "format": "date"}, "age": {"type": "integer"}, "primary_position": {"type": "string"}, "secondary_positions": {"type": "array", "items": {"type": "string"}}, "current_club_name": {"type": ["string", "null"]}, "jersey_number": {"type": ["integer", "string", "null"]}, "height_cm": {"type": ["integer", "null"]}, "weight_kg": {"type": ["integer", "null"]}, "preferred_foot": {"type": ["string", "null"], "enum": [None, "Right", "Left", "Both"]},
                   "career_summary_stats": {"type": "object", "properties": {"appearances": {"type": ["integer", "null"]}, "goals": {"type": ["integer", "null"]}, "assists": {"type": ["integer", "null"]}}},
                   "market_value": {"type": ["string", "null"]}},
    "required": ["full_name", "nationality", "date_of_birth", "primary_position"]
}
SCHEMA_DATA_TEAM_NEWS = {
    "type": "object", "title": "TeamNewsData", "description": "Latest news articles or summaries for a specific team.",
    "properties": {"team_name": {"type": "string"},
                   "news_articles": {"type": "array", "items": {"type": "object", "properties": {"title": {"type": "string"}, "source_name": {"type": ["string", "null"]}, "published_date": {"type": ["string", "null"], "format": "date-time"}, "url": {"type": ["string", "null"], "format": "uri"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}}},
    "required": ["team_name", "news_articles"]
}
# (Other schemas like Results, Team Stats, etc. would be included here as in the original file)

TOOLS_AVAILABLE = [
    {"type": "function", "function": {"name": "present_h2h_comparison", "description": "Presents a head-to-head comparison between two teams.", "parameters": SCHEMA_DATA_H2H}},
    {"type": "function", "function": {"name": "display_standings_table", "description": "Displays a league standings table.", "parameters": SCHEMA_DATA_STANDINGS_TABLE}},
    {"type": "function", "function": {"name": "show_match_schedule", "description": "Shows a schedule of upcoming matches for a specific day or period.", "parameters": SCHEMA_DATA_MATCH_SCHEDULE_TABLE}},
    {"type": "function", "function": {"name": "get_player_profile", "description": "Retrieves detailed information about a sports player.", "parameters": SCHEMA_DATA_PLAYER_PROFILE}},
    {"type": "function", "function": {"name": "get_team_news", "description": "Fetches latest news articles for a specific sports team.", "parameters": SCHEMA_DATA_TEAM_NEWS}},
    # (The full list of tools from the original file would be here)
]

TOOL_NAME_TO_COMPONENT_TYPE = {
    "present_h2h_comparison": "h2h_comparison_table",
    "display_standings_table": "standings_table",
    "show_match_schedule": "match_schedule_table",
    "get_player_profile": "player_profile_card",
    "get_team_news": "news_article_list",
    # (The full mapping from the original file would be here)
}
print("--- openai_service.py: Schemas and Tools DEFINED ---")


async def gather_real_time_data(user_query: str, conversation_history: List[Dict[str, str]]) -> str:
    """
    First Call: Uses gpt-4o with search to gather comprehensive, real-time data.
    """
    print(f"--- Step 1: GATHERING real-time data for query: '{user_query[:60]}...' ---")
    
    system_prompt = """
    You are a highly-capable Sports Information Gatherer.
    Your task is to fully understand the user's query in the context of the conversation history.
    Then, use your search capabilities to find the most relevant, accurate, and up-to-date information.
    Compile all the factual information you find—such as statistics, schedules, player details, team news, head-to-head records, or live scores—into a single, comprehensive text block.
    Do NOT format this as a chat response. Do NOT use tools. Just return the raw, gathered data.
    If the query is conversational (e.g., "hello", "who are you?", "thanks") or clearly out-of-scope (e.g., "what is the capital of France?"), state that no data fetching is required.
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history[-6:]) # Use last 3 turns for context
    messages.append({"role": "user", "content": user_query})

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-search-preview", # Using gpt-4o for its advanced reasoning and integrated search
            messages=messages,
        )
        gathered_data = response.choices[0].message.content
        print(f">>> Data gathering complete. Snippet: {gathered_data[:200]}...")
        return gathered_data
    except Exception as e:
        print(f"!!! ERROR during Step 1 (gather_real_time_data): {e}")
        traceback.print_exc()
        return f"Error: Could not gather information due to an internal error: {str(e)}"


async def generate_final_response_with_tools(
    user_query: str,
    conversation_history: List[Dict[str, str]],
    gathered_data: str
) -> Dict[str, Any]:
    """
    Second Call: Uses the gathered data to generate a friendly reply and structured UI data using tools.
    """
    print("--- Step 2: GENERATING final response and UI data ---")
    
    # Define a default response in case of failure
    final_response = {
        "reply": "I'm sorry, I had trouble processing the sports information. Please try rephrasing your request.",
        "ui_data": {"component_type": "generic_text", "data": {}}
    }

    # Handle cases where data gathering failed or was not needed
    if "no data fetching is required" in gathered_data.lower() or gathered_data.startswith("Error:"):
        # Let the model generate a conversational reply without tools
        print(">>> No data fetched or required. Generating a conversational reply.")
        system_prompt = """
        You are GameNerd, a friendly and helpful sports AI assistant.
        Based on the user's query and conversation history, provide a direct, conversational response.
        If the user is asking about you, introduce yourself.
        If the query is out of scope, politely state that you only handle sports and gaming topics.
        Do NOT use any tools. Do NOT include markdown links or URLs.
        """
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history[-6:])
        messages.append({"role": "user", "content": user_query})
    else:
        # Prepare the main prompt for data presentation and tool use
        print(">>> Data was gathered. Preparing to generate response with tools.")
        system_prompt = """
        You are GameNerd, an expert sports AI assistant.
        Your goal is to present information clearly to the user. You have been provided with a block of raw data.

        Your tasks are:
        1. Write a friendly, concise, and helpful text `reply` to the user's query based on the provided data.
        2. Analyze the provided data and the user's original request.
        3. Select the SINGLE most appropriate `tool` from the available list to structure the key information for a UI display.
        4. Populate the arguments for your chosen tool completely and accurately using the provided data.
        
        CRITICAL:
        - You MUST call a tool if the query is data-related (schedules, stats, etc.).
        - Your text `reply` must NOT contain any markdown links or URLs. Mention sources by name only if necessary.
        """
        user_content = f"""
        Here is the user's original query: "{user_query}"

        Here is the raw information I gathered for you:
        <DATA_BLOCK>
        {gathered_data}
        </DATA_BLOCK>

        Now, please perform your tasks as instructed.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS_AVAILABLE if "no data fetching" not in gathered_data.lower() else None,
            tool_choice="auto" if "no data fetching" not in gathered_data.lower() else None,
            temperature=0.2,
        )

        response_message = response.choices[0].message
        
        # Set the text reply
        if response_message.content:
            # Simple link stripping as a safeguard
            final_response["reply"] = re.sub(r'\[(.*?)\]\(http[s]?://.*?\)', r'\1', response_message.content)
        
        # Process tool call for structured UI data
        if response_message.tool_calls:
            tool_call = response_message.tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            component_type = TOOL_NAME_TO_COMPONENT_TYPE.get(function_name, "generic_text")
            final_response["ui_data"] = {
                "component_type": component_type,
                "data": function_args
            }
            # If the LLM didn't provide a text reply, create a default one
            if not response_message.content:
                final_response["reply"] = "Certainly! Here is the information you requested."
            print(f">>> UI component generated: '{component_type}'")
        else:
            print(">>> No tool call was made. Response is text-only.")
            # If no tool was called and no text was generated, use a fallback.
            if not response_message.content:
                 final_response["reply"] = "I've processed your request."

        return final_response

    except Exception as e:
        print(f"!!! UNEXPECTED ERROR in Step 2 (generate_final_response_with_tools): {e}")
        traceback.print_exc()
        # Return the default error response defined at the start of the function
        final_response["reply"] = f"An unexpected server error occurred while formatting the response: {str(e)}"
        final_response["ui_data"]["data"]["error"] = str(e)
        return final_response


async def process_user_query(user_query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Main orchestrator for the two-call process.
    """
    # Step 1: Gather data
    gathered_data = await gather_real_time_data(user_query, conversation_history)
    
    # Step 2: Generate the final response using the gathered data
    final_result = await generate_final_response_with_tools(user_query, conversation_history, gathered_data)
    
    return final_result

print("--- openai_service.py: All functions DEFINED. Ready. ---")