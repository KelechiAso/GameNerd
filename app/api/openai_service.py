# /app/api/openai_service.py
'''
import json
import openai
import os
import asyncio
import traceback
import html
import re # For link stripping fallback
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

from openai import AsyncOpenAI, Timeout

# --- Setup ---
print("--- openai_service.py: TOP OF FILE ---")
print("--- openai_service.py: About to load dotenv ---")
load_dotenv()
print("--- openai_service.py: dotenv loaded ---")

openai_api_key = os.getenv("OPENAI_API_KEY")
print(f"--- openai_service.py: OPENAI_API_KEY is {'SET' if openai_api_key else 'MISSING!'} ---")

if not openai_api_key:
    print("CRITICAL ERROR from openai_service.py: OPENAI_API_KEY environment variable not set.")
    raise ValueError("CRITICAL: OPENAI_API_KEY is not set in openai_service.py. AI Service cannot initialize.")

print("--- openai_service.py: About to initialize AsyncOpenAI client ---")
client = AsyncOpenAI(
    api_key=openai_api_key,
    timeout=Timeout(120.0, connect=10.0)
)
print("--- openai_service.py: AsyncOpenAI client INITIALIZED ---")


# --- Constants ---
# System prompt for Step 0: V3.2 - Refined request_type logic
SYSTEM_PROMPT_EXTRACTOR_V3_2 = """
Task: Analyze the **current user query** in the context of the **provided conversation history**. Extract structured sports information, classify the query, or provide a direct simple reply for acknowledgments/basic conversational turns/out-of-scope requests.

**Input Context:**
You will receive:
1.  (Optional) Recent `Conversation History`.
2.  The `Current User Query`.

**Instructions:**

1.  **Deeply Understand Context & History:**
    * Analyze `Conversation History` to understand the dialogue flow.
    * Determine if the `Current User Query` is an acknowledgment, a simple greeting, related to sports/gaming, or clearly off-topic.
    * Resolve pronouns, fill in missing entities, and understand implicit references.
    * Analyze User Tone: Determine the user's likely tone (e.g., "formal", "informal", "enthusiastic", "neutral", "frustrated").
    * Detect Language (Optional but helpful): Identify the primary language of the `Current User Query` (e.g., "English", "Nigerian Pidgin", "Yoruba"). If it's a mix, note that. Prioritize standard English if unsure about complex local dialects unless the query is predominantly in that dialect.

2.  **Output JSON:** Based on your contextual understanding, output ONLY a **JSON** object:

    * **If the query is clearly unrelated to sports, gaming, betting statistics, or your capabilities as a sports AI (e.g., asking for math help, general knowledge unrelated to sports, personal advice, medical questions)**:
        * "input_type": "out_of_scope_request"
        * "user_tone": "informal", // Extracted tone
        * "detected_language": "English",
        * "conversation": "Generate a brief, friendly, and slightly varied welcome message. Introduce yourself as GameNerd, the sports and gaming information assistant, and ask how you can help. Examples: 'Hi there! GameNerd, your sports AI, at your service. What can I do for you?', 'Hello! GameNerd here. How can I assist with your sports or gaming questions today?'"
        * "original_query": The original current user input text.
        * "contextual_query_interpretation": "User query is outside of the AI's defined scope (sports and gaming)."

    * **If the query is a simple acknowledgment (e.g., "okay", "thanks", "got it", "nice")**:
        * "input_type": "acknowledgment"
        * "user_tone": "informal", // Extracted tone
        * "detected_language": "English",
        * "conversation": "Okay."
        * "original_query": The original current user input text.
        * "contextual_query_interpretation": "User is acknowledging previous information."

    * **If the query is a simple greeting (e.g., "hi", "hello") AND the conversation is just starting or history shows no active complex topic**:
        * "input_type": "simple_greeting"
         * "user_tone": "informal", // Extracted tone
        * "detected_language": "English",
        * "conversation": "Hello! I'm GameNerd, your sports and gaming information assistant. How can I help you today?"
        * "original_query": The original current user input text.
        * "contextual_query_interpretation": "Simple user greeting."

    * **If the query asks about your identity/purpose (e.g., "who are you?") AND you haven't *just* introduced yourself (check history)**:
        * "input_type": "identity_query"
         * "user_tone": "informal", // Extracted tone
        * "detected_language": "English",
        * "conversation": ""
        * "original_query": The original current user input text.
        * "contextual_query_interpretation": "User is asking about AI's identity/purpose. Needs full reply from Step 2."

    * **If the query (with context) is a sports or gaming-related query (and not a simple acknowledgment/greeting/out_of_scope)**:
        * "input_type": "sports_query"
         * "user_tone": "informal", // Extracted tone
        * "detected_language": "English",
        * "sport_id": Numerical ID (Use mapping: {{1: Soccer, 18: Basketball, 13: Tennis, 91: Volleyball, 78: Handball, 16: Baseball, 2: Horse Racing, 17: Ice Hockey, 12: American Football, 3: Cricket}}). **Crucially identify the sport if mentioned (e.g., "basketball", "NBA", "WNBA" implies sport_id 18).** Use `null` if truly unclear even with history.
        * "teams": Array of team names. Empty array ([]) if none.
        * "request_type": String classification.
            **- If the user asks for "games today", "schedule for today", "what games are on", "fixtures today", "upcoming matches for today", "top games today", "best games today", "important games today", or similar phrases implying a list of games for the current day, classify as "match_schedule".**
            **- For general schedules not tied to a specific day but asking for a list of games (e.g., "upcoming basketball games"), also use "match_schedule".**
            **- For requests about currently playing games, use "live_score".**
            **- Other examples: ["scores", "H2H Details", "player_profile", "standings", "team_statistics", "suggestion", "percentage_request", "news"].**
        * "cc": Dictionary {{ "Team Name": "iso_code" }}. Empty object ({{}}) if none/unknown.
        * "original_query": The original current user input text.
        * "conversation": ""
        * "contextual_query_interpretation": (Optional) Brief explanation of query interpretation (e.g., "User is asking for the basketball game schedule for today.").

    * **If the query is general chit-chat related to sports/gaming, sports trivia, or a conversational follow-up that might need more than a canned reply but doesn't fit sports/identity (and isn't a simple greeting/acknowledgment/out_of_scope)**:
        * "input_type": "conversational"
         * "user_tone": "informal", // Extracted tone
        * "detected_language": "English",
        * "conversation": ""
        * "original_query": current_user_query
        * "contextual_query_interpretation": "General sports/gaming conversational query or trivia for Step 2."

    * **If the query, within the sports/gaming domain, is not recognizable or lacks sufficient information**:
        * "input_type": "invalid_request"
         * "user_tone": "informal", // Extracted tone
        * "detected_language": "English",
        * "conversation": "I'm not sure how to help with that sports or gaming query. Could you please rephrase or provide more details?"
        * "original_query": current_user_query
        * "contextual_query_interpretation": "Sports/gaming query is unclear or lacks actionable information."

3.  **Be Strict & Context-Aware:** Output ONLY the valid **JSON**.
"""

FETCH_INFO_PROMPT_TEMPLATE = """
You are a Sports Information Gatherer. Your task is to find factual information related to **sports and gaming queries ONLY**.
Based ONLY on the user's query below (and any provided context if it's a follow-up), provide a comprehensive and informative text answer using your search capabilities.
Focus on facts, statistics, schedules, H2H details, player info, or other pertinent details found that seem relevant to the **sports or gaming query**.
Present the information clearly. Do NOT include any markdown links or URLs in your output. Mention sources by name if absolutely necessary, but do not format them as links.

If the query context indicates it's not a sports or gaming query that requires data fetching (e.g., it's about AI identity, or Step 0 misclassified an out-of-scope query), state "No specific sports/gaming data fetching is required for this query."

User Query Context: "{contextual_interpretation}"
User Query: "{user_query}"
"""

# --- All SCHEMA_DATA_... definitions ---
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
SCHEMA_DATA_RESULTS_LIST = {
    "type": "object", "title": "ResultsListData", "description": "Data for a list of match results.",
    "properties": {"matches": {"type": "array", "items": {"type": "object", "properties": {
                    "date": {"type": "string"}, "time": {"type": "string"}, "competition": {"type": "string"}, "round": {"type": ["string", "null"]},
                    "home_team_name": {"type": "string"}, "away_team_name": {"type": "string"},
                    "score": {"type": "object", "properties": {"fulltime": {"type": "object", "properties": {"home": {"type": ["integer", "null"]}, "away": {"type": ["integer", "null"]}}}, "halftime": {"type": "object", "properties": {"home": {"type": ["integer", "null"]}, "away": {"type": ["integer", "null"]}}, "description": "Optional halftime score"}}, "required": ["fulltime"]},
                    "status": {"type": "string"}}, "required": ["date", "home_team_name", "away_team_name", "score", "status"]}}},
    "required": ["matches"]
}
SCHEMA_DATA_MATCH_SCHEDULE_TABLE = {
    "type": "object", "title": "MatchScheduleTableData", "description": "Data for a table of upcoming matches.",
    "properties": {"title": {"type": "string"}, "headers": {"type": "array", "items": {"type": "string"}},
                   "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                   "sort_info": {"type": ["string", "null"]}},
    "required": ["headers", "rows"]
}
SCHEMA_DATA_TEAM_STATS = {
    "type": "object", "title": "TeamStatsData", "description": "Data for team (or general) statistics.",
    "properties": {"stats_type": {"type": "string"}, "title": {"type": "string"},
                   "sections": {"type": "array", "items": {"type": "object", "properties": {"section_title": {"type": "string"}, "key_value_pairs": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean", "null"]}}}, "required": ["section_title", "key_value_pairs"]}},
                   "narrative_summary": {"type": ["string", "null"]}},
    "required": ["stats_type", "title"]
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
SCHEMA_DATA_SUGGESTION_CARD = {
    "type": "object", "title": "SuggestionCardData", "description": "Data for suggestions or predictions.",
    "properties": {"title": {"type": "string"}, "details": {"type": "string"}, "key_points": {"type": "array", "items": {"type": "string"}},
                   "confidence_level": {"type": ["string", "null"]}, "disclaimer": {"type": "string", "default": "This is based on statistical analysis and not a guarantee."}},
    "required": ["title", "details"]
}
SCHEMA_DATA_PERCENTAGE_CARD = {
    "type": "object", "title": "PercentageCardData", "description": "Data for displaying a specific percentage.",
    "properties": {"title": {"type": "string"}, "value": {"type": "string", "pattern": "^(\\d{1,3}(\\.\\d+)?%|\\d+/\\d+)$"}, "context": {"type": "string"}, "basis": {"type": ["string", "null"]},
                   "supporting_stats": {"type": "object", "additionalProperties": {"type": ["string", "number"]}}},
    "required": ["title", "value", "context"]
}
SCHEMA_DATA_LIVE_MATCH_DETAILS = {
    "type": "object", "title": "LiveMatchDetailsData", "description": "Data for a currently ongoing live match.",
    "properties": {"match_id": {"type": ["string", "null"]}, "competition": {"type": "string"}, "home_team_name": {"type": "string"}, "away_team_name": {"type": "string"},
                   "home_team_score": {"type": "integer"}, "away_team_score": {"type": "integer"}, "current_minute": {"type": "string"}, "status_description": {"type": "string"},
                   "key_events": {"type": "array", "items": {"type": "object", "properties": {"minute": {"type": "string"}, "type": {"type": "string", "enum": ["goal", "yellow_card", "red_card", "substitution", "var_decision", "penalty_scored", "penalty_missed", "own_goal"]}, "player_name": {"type": ["string", "null"]}, "team_name": {"type": "string"}, "detail": {"type": ["string", "null"]}}, "required": ["minute", "type", "team_name"]}},
                   "live_stats": {"type": "object", "properties": {"home_possession": {"type": ["integer", "null"]}, "away_possession": {"type": ["integer", "null"]}, "home_shots": {"type": ["integer", "null"]}, "away_shots": {"type": ["integer", "null"]}, "home_shots_on_target": {"type": ["integer", "null"]}, "away_shots_on_target": {"type": ["integer", "null"]}, "home_corners": {"type": ["integer", "null"]}, "away_corners": {"type": ["integer", "null"]}}}},
    "required": ["competition", "home_team_name", "away_team_name", "home_team_score", "away_team_score", "current_minute", "status_description"]
}
SCHEMA_DATA_MATCH_LINEUPS = {
    "type": "object", "title": "MatchLineupsData", "description": "Lineup information for a specific match.",
    "properties": {"match_description": {"type": "string"},
                   "home_team": {"type": "object", "properties": {"name": {"type": "string"}, "formation": {"type": ["string", "null"]}, "manager": {"type": ["string", "null"]}, "starting_xi": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}, "substitutes": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}}, "required": ["name", "starting_xi"]},
                   "away_team": {"type": "object", "properties": {"name": {"type": "string"}, "formation": {"type": ["string", "null"]}, "manager": {"type": ["string", "null"]}, "starting_xi": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}, "substitutes": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}}, "required": ["name", "starting_xi"]}},
    "required": ["match_description", "home_team", "away_team"]
}
SCHEMA_DATA_TOP_PERFORMERS = {
    "type": "object", "title": "TopPerformersData", "description": "Lists top performing players in a specific statistic for a league/tournament.",
    "properties": {"league_name": {"type": "string"}, "season": {"type": ["string", "null"]}, "statistic_type": {"type": "string"},
                   "performers": {"type": "array", "items": {"type": "object", "properties": {"rank": {"type": "integer"}, "player_name": {"type": "string"}, "team_name": {"type": "string"}, "value": {"type": ["integer", "string"]}, "nationality": {"type": ["string", "null"]}}, "required": ["rank", "player_name", "team_name", "value"]}}},
    "required": ["league_name", "statistic_type", "performers"]
}
SCHEMA_DATA_PLAYER_PROFILE = {
    "type": "object", "title": "PlayerProfileData", "description": "Detailed profile information for a specific player.",
    "properties": {"full_name": {"type": "string"}, "common_name": {"type": ["string", "null"]}, "nationality": {"type": "string"}, "date_of_birth": {"type": "string", "format": "date"}, "age": {"type": "integer"}, "primary_position": {"type": "string"}, "secondary_positions": {"type": "array", "items": {"type": "string"}}, "current_club_name": {"type": ["string", "null"]}, "jersey_number": {"type": ["integer", "string", "null"]}, "height_cm": {"type": ["integer", "null"]}, "weight_kg": {"type": ["integer", "null"]}, "preferred_foot": {"type": ["string", "null"], "enum": [None, "Right", "Left", "Both"]},
                   "career_summary_stats": {"type": "object", "properties": {"appearances": {"type": ["integer", "null"]}, "goals": {"type": ["integer", "null"]}, "assists": {"type": ["integer", "null"]}}},
                   "market_value": {"type": ["string", "null"]}},
    "required": ["full_name", "nationality", "date_of_birth", "primary_position"]
}
SCHEMA_DATA_PLAYER_COMPARISON = {
    "type": "object", "title": "PlayerComparisonData", "description": "Side-by-side comparison of statistics for two or more players.",
    "properties": {"comparison_title": {"type": "string"},
                   "players": {"type": "array", "minItems": 2, "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "team_name": {"type": ["string", "null"]}, "stats": {"type": "object", "additionalProperties": {"type": ["string", "number", "null"]}}}, "required": ["player_name", "stats"]}},
                   "comparison_period": {"type": ["string", "null"]}},
    "required": ["comparison_title", "players"]
}
SCHEMA_DATA_SPORTS_TERM_EXPLANATION = {
    "type": "object", "title": "SportsTermExplanationData", "description": "Explanation of a sports-related term or rule.",
    "properties": {"term": {"type": "string"}, "explanation": {"type": "string"}, "sport": {"type": ["string", "null"]},
                   "related_terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["term", "explanation"]
}
SCHEMA_DATA_TEAM_NEWS = {
    "type": "object", "title": "TeamNewsData", "description": "Latest news articles or summaries for a specific team.",
    "properties": {"team_name": {"type": "string"},
                   "news_articles": {"type": "array", "items": {"type": "object", "properties": {"title": {"type": "string"}, "source_name": {"type": ["string", "null"]}, "published_date": {"type": ["string", "null"], "format": "date-time"}, "url": {"type": ["string", "null"], "format": "uri"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}}},
    "required": ["team_name", "news_articles"]
}
SCHEMA_DATA_LEAGUE_NEWS = {
    "type": "object", "title": "LeagueNewsData", "description": "Latest news articles or summaries for a specific league.",
    "properties": {"league_name": {"type": "string"},
                   "news_articles": {"type": "array", "items": {"type": "object", "properties": {"title": {"type": "string"}, "source_name": {"type": ["string", "null"]}, "published_date": {"type": ["string", "null"], "format": "date-time"}, "url": {"type": ["string", "null"], "format": "uri"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}}},
    "required": ["league_name", "news_articles"]
}
print("--- openai_service.py: All SCHEMA_DATA_... DEFINED ---")

# --- Tool Definitions ---
TOOLS_AVAILABLE = [
    {"type": "function", "function": {"name": "present_h2h_comparison", "description": "Presents a head-to-head comparison between two teams.", "parameters": SCHEMA_DATA_H2H}},
    {"type": "function", "function": {"name": "display_standings_table", "description": "Displays a league standings table.", "parameters": SCHEMA_DATA_STANDINGS_TABLE}},
    {"type": "function", "function": {"name": "show_match_schedule", "description": "Shows a schedule of upcoming matches (today, future) or top games for a specific day.", "parameters": SCHEMA_DATA_MATCH_SCHEDULE_TABLE}},
    {"type": "function", "function": {"name": "list_match_results", "description": "Lists recent match results.", "parameters": SCHEMA_DATA_RESULTS_LIST}},
    {"type": "function", "function": {"name": "provide_team_statistics", "description": "Provides statistics for a team or a general statistical overview.", "parameters": SCHEMA_DATA_TEAM_STATS}},
    {"type": "function", "function": {"name": "offer_suggestion", "description": "Offers a suggestion or prediction based on data (e.g., match outcome likelihood).", "parameters": SCHEMA_DATA_SUGGESTION_CARD}},
    {"type": "function", "function": {"name": "analyze_percentage", "description": "Provides an analysis for a percentage-based query (e.g., chance of winning X%).", "parameters": SCHEMA_DATA_PERCENTAGE_CARD}},
    {"type": "function", "function": {"name": "get_live_match_details", "description": "Provides real-time details for an ongoing match (score, time, key events).", "parameters": SCHEMA_DATA_LIVE_MATCH_DETAILS}},
    {"type": "function", "function": {"name": "get_match_lineups", "description": "Provides starting lineups and substitutes for a match.", "parameters": SCHEMA_DATA_MATCH_LINEUPS}},
    {"type": "function", "function": {"name": "get_top_performers", "description": "Lists top performing players (e.g., scorers, assists) for a league/tournament.", "parameters": SCHEMA_DATA_TOP_PERFORMERS}},
    {"type": "function", "function": {"name": "get_player_profile", "description": "Retrieves detailed information about a sports player (bio, current team, basic stats).", "parameters": SCHEMA_DATA_PLAYER_PROFILE}},
    {"type": "function", "function": {"name": "compare_players", "description": "Provides a side-by-side statistical comparison of players.", "parameters": SCHEMA_DATA_PLAYER_COMPARISON}},
    {"type": "function", "function": {"name": "clarify_sports_term", "description": "Explains a specific sports term, rule, or concept.", "parameters": SCHEMA_DATA_SPORTS_TERM_EXPLANATION}},
    {"type": "function", "function": {"name": "get_team_news", "description": "Fetches latest news articles related to a specific sports team.", "parameters": SCHEMA_DATA_TEAM_NEWS}},
    {"type": "function", "function": {"name": "get_league_news", "description": "Fetches latest news articles related to a specific sports league.", "parameters": SCHEMA_DATA_LEAGUE_NEWS}},
]
print("--- openai_service.py: TOOLS_AVAILABLE DEFINED ---")

TOOL_NAME_TO_COMPONENT_TYPE = {
    "present_h2h_comparison": "h2h_comparison_table",
    "display_standings_table": "standings_table",
    "show_match_schedule": "match_schedule_table",
    "list_match_results": "results_list",
    "provide_team_statistics": "team_stats",
    "offer_suggestion": "suggestion_card",
    "analyze_percentage": "percentage_card",
    "get_live_match_details": "live_match_feed",
    "get_match_lineups": "match_lineups_display",
    "get_top_performers": "top_performers_list",
    "get_player_profile": "player_profile_card",
    "compare_players": "player_comparison_view",
    "clarify_sports_term": "term_explanation_box",
    "get_team_news": "news_article_list",
    "get_league_news": "news_article_list",
}
print("--- openai_service.py: TOOL_NAME_TO_COMPONENT_TYPE DEFINED ---")

# --- Helper Functions ---
def safeText(text: Any) -> str:
    if text is None or isinstance(text, (dict, list)): return 'N/A'
    text_str = str(text); return html.escape(text_str)
print("--- openai_service.py: safeText function DEFINED ---")

# --- AI Service Core Functions ---
async def extract_sports_info(user_query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    print(f"--- openai_service.py: extract_sports_info CALLED with query: '{user_query[:50]}...' ---")
    if not user_query: return {"input_type": "invalid_request", "conversation": "", "original_query": ""}
    fallback_error = {"input_type": "error", "message": "Extraction failed during initial processing.", "original_query": user_query}

    messages = [{"role": "system", "content": SYSTEM_PROMPT_EXTRACTOR_V3_2}]
    history_to_include = conversation_history[-4:] # Use last 4 turns for context
    for turn in history_to_include:
        content = turn.get("content", "") 
        if not isinstance(content, str): content = str(content)
        messages.append({"role": turn["role"], "content": content})
    messages.append({"role": "user", "content": user_query})

    try:
        print(f">>> Attempting extraction call (GPT-3.5 Turbo, JSON Mode, V3.2 Prompt)...")
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1 # Lower temperature for more deterministic classification
        )
        extracted_data = json.loads(response.choices[0].message.content)
        
        extracted_data["original_query"] = extracted_data.get("original_query", user_query)
        extracted_data["input_type"] = extracted_data.get("input_type", "unknown")
        
        # Ensure 'conversation' field is appropriately set or defaulted for types that might be short-circuited
        direct_reply_types_with_predefined_conv = ["acknowledgment", "simple_greeting", "invalid_request", "out_of_scope_request"]
        types_needing_empty_conv_for_step2 = ["identity_query", "sports_query", "conversational"]

        if extracted_data["input_type"] in direct_reply_types_with_predefined_conv:
            if "conversation" not in extracted_data or not extracted_data.get("conversation","").strip():
                # The prompt SYSTEM_PROMPT_EXTRACTOR_V3_2 should already provide these. This is a fallback.
                default_conversations = {
                    "simple_greeting": "Hello! I'm GameNerd, your sports and gaming information assistant. How can I help you today?",
                    "invalid_request": "I'm not sure how to help with that sports or gaming query. Could you please rephrase or provide more details?",
                    "out_of_scope_request": "I am GameNerd, an AI assistant focused on sports and gaming information. I'm unable to assist with requests outside of this domain.",
                    "acknowledgment": "Okay."
                }
                extracted_data["conversation"] = default_conversations.get(extracted_data["input_type"], "Processing your request.")
        elif extracted_data["input_type"] in types_needing_empty_conv_for_step2:
             extracted_data["conversation"] = "" # Ensure it's empty if Step 2 needs to generate it

        print(f"Extracted data from Step 0: {json.dumps(extracted_data, indent=2)}")
        return extracted_data
    except json.JSONDecodeError as json_err:
        print(f"!!! JSON decoding error during extraction: {json_err}")
        raw_output_content = "Error fetching raw output for JSONDecodeError"
        try: # Try to get the raw response if possible
            if 'response' in locals() and response.choices and response.choices[0].message:
                raw_output_content = response.choices[0].message.content
        except Exception: pass
        print(f"Raw LLM output (extraction attempt): {raw_output_content}")
        fallback_error["message"] = "Failed to parse AI's initial analysis."
        return fallback_error
    except openai.APIError as api_err:
        print(f"!!! OPENAI API ERROR during extraction !!! Error details: {api_err}")
        fallback_error["message"] = f"OpenAI API Error during extraction: {str(api_err)}"
        return fallback_error
    except Exception as e:
        print(f"!!! An error occurred during extraction: {e}")
        traceback.print_exc()
        fallback_error["message"] = f"Unexpected error during query extraction: {str(e)}"
        return fallback_error

async def fetch_raw_text_data(user_query: str, parsed_data: Optional[Dict[str, Any]], conversation_history: List[Dict[str, str]]) -> str:
    print(f"--- openai_service.py: fetch_raw_text_data CALLED for input_type: {parsed_data.get('input_type')} ---")
    input_type = parsed_data.get("input_type") if parsed_data else "unknown"
    
    no_fetch_types = ["simple_greeting", "acknowledgment", "invalid_request", "identity_query", "out_of_scope_request"]
    request_type_from_step0 = parsed_data.get("request_type") if parsed_data else None

    if input_type in no_fetch_types:
         print(f">>> Skipping raw text data fetch: Input type '{input_type}' does not require fetching.")
         return "--NO DATA FETCHED (Input Type Not Requiring Fetch)--"
    
    requires_fetch = False
    if input_type == "sports_query":
        requires_fetch = True
    elif input_type == "conversational" and "trivia" in parsed_data.get("contextual_query_interpretation","").lower(): # Fetch for trivia
        requires_fetch = True
    # Explicit request_types that inherently need data, even if input_type was classified broadly as 'conversational' by mistake
    data_needing_request_types = [
        "scores", "H2H Details", "player_profile", "live_score", "standings", 
        "team_statistics", "suggestion", "percentage_request", "news", "match_schedule",
        "top_performers", "match_lineups", "compare_players" # Added new ones
    ]
    if request_type_from_step0 in data_needing_request_types:
        requires_fetch = True

    if not requires_fetch:
        print(f">>> Skipping raw text data fetch: Input type '{input_type}' and request type '{request_type_from_step0}' do not necessitate fetching.")
        return "--NO DATA FETCHED (Logic determined no fetch needed)--"

    if not parsed_data: # Should be caught by requires_fetch logic if it depends on parsed_data, but as a safeguard
        print(">>> Skipping raw text data fetch: Parsed data is missing (should not happen if fetch is required).")
        return "--NO DATA FETCHED (Missing Parsed Data Unexpectedly)--"

    query_for_fetch = parsed_data.get("original_query", user_query)
    contextual_interpretation = parsed_data.get("contextual_query_interpretation", "")
    
    effective_query_for_fetch = query_for_fetch
    if contextual_interpretation and contextual_interpretation.strip() and "User asked:" not in contextual_interpretation :
        # More robust check to avoid simple prefix repetition
        if len(contextual_interpretation) > 5 and not effective_query_for_fetch.lower().startswith(contextual_interpretation.lower()[:len(effective_query_for_fetch)-10]):
             effective_query_for_fetch = f"Context from prior analysis: {contextual_interpretation}\nUser's current query: {query_for_fetch}"
    
    try:
        prompt = FETCH_INFO_PROMPT_TEMPLATE.format(
            user_query=query_for_fetch, 
            contextual_interpretation=contextual_interpretation if contextual_interpretation.strip() else "No specific prior context."
        )
    except Exception as fmt_e:
         print(f"!!! UNEXPECTED ERROR during data fetch prompt formatting !!! Error: {fmt_e}")
         return f"--PROMPT FORMATTING ERROR: {fmt_e}--"

    messages_for_fetch = [{"role": "system", "content": prompt}]

    try:
        print(f">>> Attempting Search LLM call for raw text data. Effective Query: {effective_query_for_fetch[:150]}...")
        response = await client.chat.completions.create(
            model="gpt-4o-search-preview", 
            messages=messages_for_fetch,
            #temperature=0.2 # Factual retrieval
        )
        raw_data_string = response.choices[0].message.content.strip()
        print(f">>> Raw Text Data Received (first 300 chars):\n----------\n{raw_data_string[:300]}...\n----------")
        
        # Check for more nuanced non-committal fetch responses
        non_committal_phrases = [
            "sorry", "don't have specific information", "cannot provide", "unable to find",
            "No specific sports/gaming data fetching is required", "cannot assist with that topic"
        ]
        if not raw_data_string or \
           (len(raw_data_string) < 100 and any(phrase in raw_data_string.lower() for phrase in non_committal_phrases)):
            print(f">>> Fetch seemed non-committal or empty: {raw_data_string}")
            return f"--FETCH FAILED OR REFUSED: {raw_data_string}--"
        return raw_data_string
    except openai.APIError as api_err:
        print(f"!!! OPENAI API ERROR during raw text data fetch !!! Error details: {api_err}")
        return f"--API ERROR FETCHING DATA: {str(api_err)}--"
    except Exception as e:
        print(f"!!! UNEXPECTED EXCEPTION during raw text data fetch !!! Type: {type(e)}, Details: {e}")
        traceback.print_exc()
        return f"--UNEXPECTED ERROR FETCHING DATA: {str(e)}--"

async def get_structured_data_and_reply_via_tools(
    parsed_data: Dict[str, Any],
    conversation_history: List[Dict[str, str]],
    raw_text_data: str
) -> Dict[str, Any]:
    print(f"--- openai_service.py: get_structured_data_and_reply_via_tools CALLED ---")
    original_query = parsed_data.get("original_query", "N/A")
    contextual_interpretation = parsed_data.get("contextual_query_interpretation", "No specific interpretation from Step 0.")
    input_type_step0 = parsed_data.get("input_type", "unknown")
    request_type_step0 = parsed_data.get("request_type", "N/A") # This is key for guiding tool selection

    tool_system_prompt = """
    You are GameNerd, a specialized AI assistant for **sports and gaming information ONLY**.
    Your primary goal is to understand the user's sports/gaming query, use relevant information, and then select the MOST appropriate tool to structure and present the data.
    
    CRITICALLY IMPORTANT: Your textual reply MUST NOT contain any markdown links (e.g., `[text](URL)`), full URLs, or hyperlinks. If you need to mention a source or entity, refer to it by its name ONLY.

    Process:
    1. If the query is clearly outside your sports/gaming domain (based on context passed), politely state: "I am GameNerd, an AI assistant focused on sports and gaming information. I'm unable to assist with requests outside of this domain." Do NOT use tools or attempt to answer.
    2. If the query is within your sports/gaming domain:
        a. Based on the 'Detected Request Type' and 'Information Gathered', decide if a tool is best.
        b. If a tool is appropriate (e.g., 'match_schedule', 'standings_table', 'player_profile'), call it with accurately extracted arguments from the 'Information Gathered' or query.
        c. Formulate a concise, user-facing textual reply. If you use a tool, this reply should briefly introduce or summarize the data.
        d. If no tool is suitable but it's a valid sports/gaming conversational topic or trivia, provide a direct textual reply.
    Ensure all required fields for a chosen tool's parameters are populated. If crucial information for a tool is missing from the gathered text, do not call the tool; instead, provide a text reply explaining what information you found or what's missing.
    """
    
    messages = [{"role": "system", "content": tool_system_prompt}]
    
    # Add some conversation history for context
    for turn in conversation_history[-3:]: # Last 3 turns
        messages.append({"role": turn["role"], "content": turn.get("content","")}) # Add default for content

    user_context_for_tool_call = f"""
    User Query: {original_query}
    Initial Analysis (from Step 0):
      Input Type: '{input_type_step0}'
      Sport ID: '{parsed_data.get("sport_id", "Not specified")}' 
      Detected Request Type: '{request_type_step0}'
      Contextual Interpretation: '{contextual_interpretation}'
    
    Information Gathered (from Step 1, if any):
    ```text
    {raw_text_data if raw_text_data and not raw_text_data.startswith("--") else "No specific data was fetched, or the query was not data-oriented (e.g., conversational, identity)."}
    ```
    Proceed according to your role and instructions. If 'Detected Request Type' is 'match_schedule', prioritize using the 'show_match_schedule' tool. If 'standings', use 'display_standings_table', etc. If 'identity_query', introduce yourself as GameNerd. Remember: NO LINKS in your textual reply.
    """
    messages.append({"role": "user", "content": user_context_for_tool_call})

    # Default response if things go wrong before LLM call or if LLM fails to respond
    final_reply = "I'm having a bit of trouble with that sports/gaming request. Could you try rephrasing?"
    ui_data = {"component_type": "generic_text", "data": {"message": "Could not generate specific UI data for this request."}}

    if input_type_step0 == "out_of_scope_request": # This check is important
        return {
            "reply": parsed_data.get("conversation", "I specialize in sports and gaming information and cannot assist with that topic."),
            "ui_data": {"component_type": "generic_text", "data": {"message": "Request outside of sports/gaming domain."}}
        }

    try:
        print(f">>> Attempting LLM call with Tools (Model: gpt-4o). Input Type: {input_type_step0}, Request Type: {request_type_step0}")
        response = await client.chat.completions.create(
            model="gpt-4o", 
            messages=messages,
            tools=TOOLS_AVAILABLE,
            tool_choice="auto", 
            temperature=0.1 # Very low temperature for precise tool use and factual replies
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if response_message.content: # LLM provided a direct textual reply
            final_reply = response_message.content
            print(f">>> LLM provided direct reply content: \"{final_reply[:150]}...\"")
        
        if tool_calls:
            print(f">>> Tool calls received: {len(tool_calls)}. Processing the first one.")
            tool_call = tool_calls[0] 
            function_name = tool_call.function.name
            function_args_json = tool_call.function.arguments
            
            print(f"  Tool to call: {function_name}")
            print(f"  Tool arguments (raw JSON string): {function_args_json[:300]}...")

            try:
                function_args = json.loads(function_args_json)
                ui_data["component_type"] = TOOL_NAME_TO_COMPONENT_TYPE.get(function_name, function_name)
                ui_data["data"] = function_args 

                if not response_message.content and ui_data["component_type"] != "generic_text": 
                    component_name_readable = ui_data['component_type'].replace('_', ' ').title()
                    final_reply = f"Certainly! Here is the {component_name_readable} information:"
                elif not response_message.content: # No tool and no direct reply, but not an error
                     final_reply = "I've processed that for you."
                print(f"  Successfully parsed arguments for tool {function_name}.")

            except json.JSONDecodeError as json_decode_err:
                print(f"!!! ERROR: Failed to parse tool arguments JSON for {function_name}: {json_decode_err}")
                print(f"  Faulty JSON string from LLM: {function_args_json}")
                # Use the existing final_reply if LLM gave one, otherwise set error message
                final_reply = final_reply if response_message.content else "I found information, but had an issue structuring it for display."
                ui_data = {"component_type": "generic_text", "data": {"error": f"AI failed to structure data for {function_name}.", "raw_args": function_args_json}}
        
        elif not response_message.content: # No tool call and no direct reply from LLM
            print("!!! LLM made no tool call and provided no direct textual reply content.")
            if input_type_step0 == "identity_query":
                final_reply = "I am GameNerd, your AI assistant for sports and gaming! Ask me about scores, stats, schedules, and more."
            elif raw_text_data and (raw_text_data.startswith("--FETCH FAILED") or raw_text_data.startswith("--NO DATA")):
                final_reply = "I couldn't retrieve specific information for your sports/gaming query right now."
            else: # General fallback if everything else failed to produce a reply
                final_reply = "I've processed your sports/gaming query. What else can I help you with?"
        
        # Final link stripping safeguard (though prompt should handle it)
        if final_reply and re.search(r"\[.*?\]\(http[s]?://.*?\)|http[s]?://\S+", final_reply):
            print(f"!!! WARNING: Final reply still contained links. Original: '{final_reply[:100]}...' Stripping.")
            stripped_reply = re.sub(r"\[(.*?)\]\(http[s]?://.*?\)", r"\1 (link removed)", final_reply)
            stripped_reply = re.sub(r"http[s]?://\S+", "https://www.youtube.com/watch?v=lOeQUwdAjE0&pp=0gcJCdgAo7VqN5tD", stripped_reply)
            if stripped_reply != final_reply: # Only append note if changes were made
                final_reply = stripped_reply + " (Note: Links are not displayed)."


        return {"reply": final_reply, "ui_data": ui_data}

    except openai.APIError as api_err:
        print(f"!!! OPENAI API ERROR during Tool Calling step !!! Error Details: {api_err}")
        error_message = f"An API issue occurred ({api_err.code}) while getting sports/gaming details. Please try later."
        if hasattr(api_err, 'message') and ("invalid_api_key" in api_err.message.lower() or "incorrect_api_key" in api_err.message.lower()):
            error_message = "There seems to be an issue with the API configuration. Please notify support."
        return {"reply": error_message, 
                "ui_data": {"component_type": "generic_text", "data": {"error": f"OpenAI API Error: {str(api_err)}"}}}
    except Exception as e:
        print(f"!!! UNEXPECTED EXCEPTION during Tool Calling or final processing !!! Type: {type(e)}, Details: {e}")
        traceback.print_exc()
        return {"reply": "An unexpected internal issue occurred with your sports/gaming request. Please try again.", 
                "ui_data": {"component_type": "generic_text", "data": {"error": f"Unexpected server error: {str(e)}"}}}

print("--- openai_service.py: All async functions DEFINED ---")
print("--- openai_service.py: BOTTOM OF FILE ---")
'''

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
print("--- openai_service.py: TOP OF FILE (Single-Call Architecture) ---")
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("CRITICAL: OPENAI_API_KEY is not set.")

client = AsyncOpenAI(
    api_key=openai_api_key,
    timeout=Timeout(120.0, connect=10.0)
)
print("--- openai_service.py: AsyncOpenAI client INITIALIZED ---")


# --- System Prompt ---
# This single prompt guides gpt-4o to do everything in one step.
SYSTEM_PROMPT = """
You are GameNerd, an expert AI assistant for sports and gaming information. Your primary goal is to provide accurate answers and structured data.

**Your Process:**

1.  **Analyze the Query:** Deeply understand the user's query and conversation history. The user may speak in English, Nigerian Pidgin, Yoruba, or other languages. Use your internal knowledge and search capabilities to find the relevant information.

2.  **Determine Response Type:**
    * **For Conversational Queries** (greetings, thanks, out-of-scope questions, identity): Provide a direct, friendly text reply. DO NOT use a tool. For out-of-scope, state you are a sports AI.
    * **For Data Queries** (schedules, stats, H2H, player info, news, etc.): You MUST use the MOST appropriate function tool from the provided list to structure the data you've found. You must also write a brief, user-facing text reply to introduce the data.

**Crucial Rules:**
* You must use the provided tools to structure data. This is not optional for data requests.
* Your final text reply must NOT contain any markdown links or URLs.
"""

# --- Schemas & Tools (Essential for reliable structured output) ---
# NOTE: These schemas are NOT "hardcoded prompts". They are the contract that guarantees
# gpt-4o's JSON output will be valid and usable by the frontend.
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
SCHEMA_DATA_RESULTS_LIST = {
    "type": "object", "title": "ResultsListData", "description": "Data for a list of match results.",
    "properties": {"matches": {"type": "array", "items": {"type": "object", "properties": {
                    "date": {"type": "string"}, "time": {"type": "string"}, "competition": {"type": "string"}, "round": {"type": ["string", "null"]},
                    "home_team_name": {"type": "string"}, "away_team_name": {"type": "string"},
                    "score": {"type": "object", "properties": {"fulltime": {"type": "object", "properties": {"home": {"type": ["integer", "null"]}, "away": {"type": ["integer", "null"]}}}, "halftime": {"type": "object", "properties": {"home": {"type": ["integer", "null"]}, "away": {"type": ["integer", "null"]}}, "description": "Optional halftime score"}}, "required": ["fulltime"]},
                    "status": {"type": "string"}}, "required": ["date", "home_team_name", "away_team_name", "score", "status"]}}},
    "required": ["matches"]
}
SCHEMA_DATA_MATCH_SCHEDULE_TABLE = {
    "type": "object", "title": "MatchScheduleTableData", "description": "Data for a table of upcoming matches.",
    "properties": {"title": {"type": "string"}, "headers": {"type": "array", "items": {"type": "string"}},
                   "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                   "sort_info": {"type": ["string", "null"]}},
    "required": ["headers", "rows"]
}
SCHEMA_DATA_TEAM_STATS = {
    "type": "object", "title": "TeamStatsData", "description": "Data for team (or general) statistics.",
    "properties": {"stats_type": {"type": "string"}, "title": {"type": "string"},
                   "sections": {"type": "array", "items": {"type": "object", "properties": {"section_title": {"type": "string"}, "key_value_pairs": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean", "null"]}}}, "required": ["section_title", "key_value_pairs"]}},
                   "narrative_summary": {"type": ["string", "null"]}},
    "required": ["stats_type", "title"]
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
SCHEMA_DATA_SUGGESTION_CARD = {
    "type": "object", "title": "SuggestionCardData", "description": "Data for suggestions or predictions.",
    "properties": {"title": {"type": "string"}, "details": {"type": "string"}, "key_points": {"type": "array", "items": {"type": "string"}},
                   "confidence_level": {"type": ["string", "null"]}, "disclaimer": {"type": "string", "default": "This is based on statistical analysis and not a guarantee."}},
    "required": ["title", "details"]
}
SCHEMA_DATA_PERCENTAGE_CARD = {
    "type": "object", "title": "PercentageCardData", "description": "Data for displaying a specific percentage.",
    "properties": {"title": {"type": "string"}, "value": {"type": "string", "pattern": "^(\\d{1,3}(\\.\\d+)?%|\\d+/\\d+)$"}, "context": {"type": "string"}, "basis": {"type": ["string", "null"]},
                   "supporting_stats": {"type": "object", "additionalProperties": {"type": ["string", "number"]}}},
    "required": ["title", "value", "context"]
}
SCHEMA_DATA_LIVE_MATCH_DETAILS = {
    "type": "object", "title": "LiveMatchDetailsData", "description": "Data for a currently ongoing live match.",
    "properties": {"match_id": {"type": ["string", "null"]}, "competition": {"type": "string"}, "home_team_name": {"type": "string"}, "away_team_name": {"type": "string"},
                   "home_team_score": {"type": "integer"}, "away_team_score": {"type": "integer"}, "current_minute": {"type": "string"}, "status_description": {"type": "string"},
                   "key_events": {"type": "array", "items": {"type": "object", "properties": {"minute": {"type": "string"}, "type": {"type": "string", "enum": ["goal", "yellow_card", "red_card", "substitution", "var_decision", "penalty_scored", "penalty_missed", "own_goal"]}, "player_name": {"type": ["string", "null"]}, "team_name": {"type": "string"}, "detail": {"type": ["string", "null"]}}, "required": ["minute", "type", "team_name"]}},
                   "live_stats": {"type": "object", "properties": {"home_possession": {"type": ["integer", "null"]}, "away_possession": {"type": ["integer", "null"]}, "home_shots": {"type": ["integer", "null"]}, "away_shots": {"type": ["integer", "null"]}, "home_shots_on_target": {"type": ["integer", "null"]}, "away_shots_on_target": {"type": ["integer", "null"]}, "home_corners": {"type": ["integer", "null"]}, "away_corners": {"type": ["integer", "null"]}}}},
    "required": ["competition", "home_team_name", "away_team_name", "home_team_score", "away_team_score", "current_minute", "status_description"]
}
SCHEMA_DATA_MATCH_LINEUPS = {
    "type": "object", "title": "MatchLineupsData", "description": "Lineup information for a specific match.",
    "properties": {"match_description": {"type": "string"},
                   "home_team": {"type": "object", "properties": {"name": {"type": "string"}, "formation": {"type": ["string", "null"]}, "manager": {"type": ["string", "null"]}, "starting_xi": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}, "substitutes": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}}, "required": ["name", "starting_xi"]},
                   "away_team": {"type": "object", "properties": {"name": {"type": "string"}, "formation": {"type": ["string", "null"]}, "manager": {"type": ["string", "null"]}, "starting_xi": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}, "substitutes": {"type": "array", "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "jersey_number": {"type": ["integer", "string", "null"]}, "position": {"type": ["string", "null"]}}}}}, "required": ["name", "starting_xi"]}},
    "required": ["match_description", "home_team", "away_team"]
}
SCHEMA_DATA_TOP_PERFORMERS = {
    "type": "object", "title": "TopPerformersData", "description": "Lists top performing players in a specific statistic for a league/tournament.",
    "properties": {"league_name": {"type": "string"}, "season": {"type": ["string", "null"]}, "statistic_type": {"type": "string"},
                   "performers": {"type": "array", "items": {"type": "object", "properties": {"rank": {"type": "integer"}, "player_name": {"type": "string"}, "team_name": {"type": "string"}, "value": {"type": ["integer", "string"]}, "nationality": {"type": ["string", "null"]}}, "required": ["rank", "player_name", "team_name", "value"]}}},
    "required": ["league_name", "statistic_type", "performers"]
}
SCHEMA_DATA_PLAYER_PROFILE = {
    "type": "object", "title": "PlayerProfileData", "description": "Detailed profile information for a specific player.",
    "properties": {"full_name": {"type": "string"}, "common_name": {"type": ["string", "null"]}, "nationality": {"type": "string"}, "date_of_birth": {"type": "string", "format": "date"}, "age": {"type": "integer"}, "primary_position": {"type": "string"}, "secondary_positions": {"type": "array", "items": {"type": "string"}}, "current_club_name": {"type": ["string", "null"]}, "jersey_number": {"type": ["integer", "string", "null"]}, "height_cm": {"type": ["integer", "null"]}, "weight_kg": {"type": ["integer", "null"]}, "preferred_foot": {"type": ["string", "null"], "enum": [None, "Right", "Left", "Both"]},
                   "career_summary_stats": {"type": "object", "properties": {"appearances": {"type": ["integer", "null"]}, "goals": {"type": ["integer", "null"]}, "assists": {"type": ["integer", "null"]}}},
                   "market_value": {"type": ["string", "null"]}},
    "required": ["full_name", "nationality", "date_of_birth", "primary_position"]
}
SCHEMA_DATA_PLAYER_COMPARISON = {
    "type": "object", "title": "PlayerComparisonData", "description": "Side-by-side comparison of statistics for two or more players.",
    "properties": {"comparison_title": {"type": "string"},
                   "players": {"type": "array", "minItems": 2, "items": {"type": "object", "properties": {"player_name": {"type": "string"}, "team_name": {"type": ["string", "null"]}, "stats": {"type": "object", "additionalProperties": {"type": ["string", "number", "null"]}}}, "required": ["player_name", "stats"]}},
                   "comparison_period": {"type": ["string", "null"]}},
    "required": ["comparison_title", "players"]
}
SCHEMA_DATA_SPORTS_TERM_EXPLANATION = {
    "type": "object", "title": "SportsTermExplanationData", "description": "Explanation of a sports-related term or rule.",
    "properties": {"term": {"type": "string"}, "explanation": {"type": "string"}, "sport": {"type": ["string", "null"]},
                   "related_terms": {"type": "array", "items": {"type": "string"}}},
    "required": ["term", "explanation"]
}
SCHEMA_DATA_TEAM_NEWS = {
    "type": "object", "title": "TeamNewsData", "description": "Latest news articles or summaries for a specific team.",
    "properties": {"team_name": {"type": "string"},
                   "news_articles": {"type": "array", "items": {"type": "object", "properties": {"title": {"type": "string"}, "source_name": {"type": ["string", "null"]}, "published_date": {"type": ["string", "null"], "format": "date-time"}, "url": {"type": ["string", "null"], "format": "uri"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}}},
    "required": ["team_name", "news_articles"]
}
SCHEMA_DATA_LEAGUE_NEWS = {
    "type": "object", "title": "LeagueNewsData", "description": "Latest news articles or summaries for a specific league.",
    "properties": {"league_name": {"type": "string"},
                   "news_articles": {"type": "array", "items": {"type": "object", "properties": {"title": {"type": "string"}, "source_name": {"type": ["string", "null"]}, "published_date": {"type": ["string", "null"], "format": "date-time"}, "url": {"type": ["string", "null"], "format": "uri"}, "summary": {"type": "string"}}, "required": ["title", "summary"]}}},
    "required": ["league_name", "news_articles"]
}

TOOLS_AVAILABLE = [
    {"type": "function", "function": {"name": "present_h2h_comparison", "description": "Presents a head-to-head comparison between two teams.", "parameters": SCHEMA_DATA_H2H}},
    {"type": "function", "function": {"name": "display_standings_table", "description": "Displays a league standings table.", "parameters": SCHEMA_DATA_STANDINGS_TABLE}},
    {"type": "function", "function": {"name": "show_match_schedule", "description": "Shows a schedule of upcoming matches (today, future) or top games for a specific day.", "parameters": SCHEMA_DATA_MATCH_SCHEDULE_TABLE}},
    {"type": "function", "function": {"name": "list_match_results", "description": "Lists recent match results.", "parameters": SCHEMA_DATA_RESULTS_LIST}},
    {"type": "function", "function": {"name": "provide_team_statistics", "description": "Provides statistics for a team or a general statistical overview.", "parameters": SCHEMA_DATA_TEAM_STATS}},
    {"type": "function", "function": {"name": "offer_suggestion", "description": "Offers a suggestion or prediction based on data (e.g., match outcome likelihood).", "parameters": SCHEMA_DATA_SUGGESTION_CARD}},
    {"type": "function", "function": {"name": "analyze_percentage", "description": "Provides an analysis for a percentage-based query (e.g., chance of winning X%).", "parameters": SCHEMA_DATA_PERCENTAGE_CARD}},
    {"type": "function", "function": {"name": "get_live_match_details", "description": "Provides real-time details for an ongoing match (score, time, key events).", "parameters": SCHEMA_DATA_LIVE_MATCH_DETAILS}},
    {"type": "function", "function": {"name": "get_match_lineups", "description": "Provides starting lineups and substitutes for a match.", "parameters": SCHEMA_DATA_MATCH_LINEUPS}},
    {"type": "function", "function": {"name": "get_top_performers", "description": "Lists top performing players (e.g., scorers, assists) for a league/tournament.", "parameters": SCHEMA_DATA_TOP_PERFORMERS}},
    {"type": "function", "function": {"name": "get_player_profile", "description": "Retrieves detailed information about a sports player (bio, current team, basic stats).", "parameters": SCHEMA_DATA_PLAYER_PROFILE}},
    {"type": "function", "function": {"name": "compare_players", "description": "Provides a side-by-side statistical comparison of players.", "parameters": SCHEMA_DATA_PLAYER_COMPARISON}},
    {"type": "function", "function": {"name": "clarify_sports_term", "description": "Explains a specific sports term, rule, or concept.", "parameters": SCHEMA_DATA_SPORTS_TERM_EXPLANATION}},
    {"type": "function", "function": {"name": "get_team_news", "description": "Fetches latest news articles related to a specific sports team.", "parameters": SCHEMA_DATA_TEAM_NEWS}},
    {"type": "function", "function": {"name": "get_league_news", "description": "Fetches latest news articles related to a specific sports league.", "parameters": SCHEMA_DATA_LEAGUE_NEWS}},
]

TOOL_NAME_TO_COMPONENT_TYPE = {
    "present_h2h_comparison": "h2h_comparison_table",
    "display_standings_table": "standings_table",
    "show_match_schedule": "match_schedule_table",
    "list_match_results": "results_list",
    "provide_team_statistics": "team_stats",
    "offer_suggestion": "suggestion_card",
    "analyze_percentage": "percentage_card",
    "get_live_match_details": "live_match_feed",
    "get_match_lineups": "match_lineups_display",
    "get_top_performers": "top_performers_list",
    "get_player_profile": "player_profile_card",
    "compare_players": "player_comparison_view",
    "clarify_sports_term": "term_explanation_box",
    "get_team_news": "news_article_list",
    "get_league_news": "news_article_list",
}

def strip_links(text: str) -> str:
    if not text or not isinstance(text, str): return text
    stripped = re.sub(r'\[(.*?)\]\(http[s]?://.*?\)', r'\1', text)
    stripped = re.sub(r'http[s]?://\S+', '', stripped)
    return stripped.replace('()', '').strip()

async def process_user_query(user_query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """Processes the user query in a single, efficient API call."""
    print(f"--- Processing query with single-call architecture: '{user_query[:50]}...' ---")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(conversation_history[-8:])  # Use last 4 user/assistant turns
    messages.append({"role": "user", "content": user_query})

    final_response = {
        "reply": "Sorry, I encountered an issue processing your request. Please try again.",
        "ui_data": {"component_type": "generic_text", "data": {}}
    }

    try:
        print(">>> Calling gpt-4o with tools...")
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=TOOLS_AVAILABLE,
            tool_choice="auto",
            temperature=0.1,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        # Set the text reply
        if response_message.content:
            final_response["reply"] = strip_links(response_message.content)
        else:
            final_response["reply"] = "Here is the information you requested." if tool_calls else "Request processed."
        print(f">>> Reply text: \"{final_response['reply'][:100]}...\"")

        # Process tool call for structured UI data
        if tool_calls:
            print(f">>> Tool call received: {tool_calls[0].function.name}")
            tool_call = tool_calls[0]
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            component_type = TOOL_NAME_TO_COMPONENT_TYPE.get(function_name, "generic_text")
            final_response["ui_data"] = {
                "component_type": component_type,
                "data": function_args
            }
            print(f">>> UI component generated: '{component_type}'")
        else:
            print(">>> No tool call was made. Response is text-only.")

        return final_response

    except Exception as e:
        print(f"!!! UNEXPECTED ERROR in process_user_query: {e}")
        traceback.print_exc()
        final_response["reply"] = "An unexpected server error occurred. Please try again later."
        final_response["ui_data"]["data"]["error"] = str(e)
        return final_response