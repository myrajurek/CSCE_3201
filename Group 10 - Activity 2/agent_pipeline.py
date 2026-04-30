# =============================================================================
# agent_pipeline.py - Task 3 (Metacritic Agent Orchestrator)
#
# Uses the Azure OpenAI Chat Completions API.
# No Assistant ID or azure-ai-projects SDK needed.
# =============================================================================

import json
import joblib
import pandas as pd
from openai import AzureOpenAI

# --- CONFIGURATION ---
AZURE_ENDPOINT = "https://metacritic-agent-phase2-resource.openai.azure.com/"
API_KEY        = "u6PAcHqxM8O5DfaGxEgRjTPA7aGdATn4WOH0psAQy0PBRtfu3Pk4JQQJ99CDACHYHv6XJ3w3AAAAACOGTyKH"
API_VERSION    = "2024-12-01-preview"
MODEL          = "gpt-4.1-mini"

# Agent behaviour defined here instead of in the Azure portal
SYSTEM_PROMPT = """You are a Metacritic score prediction assistant.
Your job is to collect exactly three pieces of information from the user:
  1. userscore  - a number between 0 and 10
  2. platform   - the gaming platform (e.g. PlayStation 4, PC, Xbox One, iOS)
  3. genres     - one or more genres (e.g. Action, RPG, Puzzle, Strategy)

If any of the three are missing, ask only for the missing one(s).
Once you have all three, respond with ONLY this exact format and nothing else:
EXTRACT_FEATURES_JSON {"userscore": <float>, "platform": "<str>", "genres": ["<str>", ...]}
"""


# ---------------------------------------------------------------------------
# Load Phase 1 artifacts
# ---------------------------------------------------------------------------

def load_phase1_artifacts():
    model        = joblib.load("model.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    print(f"[Phase 1] Model and {len(feature_cols)} feature columns loaded.")
    return model, feature_cols


# ---------------------------------------------------------------------------
# Encode inputs and run the Phase 1 regression model
# ---------------------------------------------------------------------------

def encode_and_predict(userscore, platform, genres, model, feature_cols):
    row = {col: 0 for col in feature_cols}

    if "userscore" in row:
        row["userscore"] = float(userscore)

    p_col = f"platforms_{platform}"
    if p_col in row:
        row[p_col] = 1

    for g in genres:
        g = g.strip()
        if g in row:
            row[g] = 1

    X         = pd.DataFrame([row])[feature_cols]
    raw       = float(model.predict(X)[0])
    predicted = round(max(0.0, min(100.0, raw)))

    if predicted >= 90:   cat = "Universal Acclaim (90-100)"
    elif predicted >= 75: cat = "Generally Favorable (75-89)"
    elif predicted >= 50: cat = "Mixed or Average (50-74)"
    else:                 cat = "Overwhelming Dislike (0-49)"

    return {"score": predicted, "category": cat}


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def run_conversation(client, model, feature_cols):
    EXTRACTION_TRIGGER = "EXTRACT_FEATURES_JSON"
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("\n[System] Metacritic Predictor Active. Type 'quit' to exit.")
    print("=" * 65)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("[System] Goodbye!")
            break

        history.append({"role": "user", "content": user_input})

        print("[Azure] Processing...", end="\r")
        response    = client.chat.completions.create(model=MODEL, messages=history)
        agent_reply = response.choices[0].message.content
        history.append({"role": "assistant", "content": agent_reply})

        # Branch 1: all inputs collected — run the prediction pipeline
        if EXTRACTION_TRIGGER in agent_reply:
            print("[Pipeline] Data captured. Calculating prediction...")
            try:
                json_start = agent_reply.index("{")
                json_end   = agent_reply.rindex("}") + 1
                data       = json.loads(agent_reply[json_start:json_end])

                result = encode_and_predict(
                    data["userscore"], data["platform"], data["genres"],
                    model, feature_cols,
                )

                explanation_prompt = (
                    f"The regression model predicted a metascore of {result['score']} "
                    f"({result['category']}). Give the user a brief, friendly "
                    f"explanation of what this means for their game."
                )
                history.append({"role": "user", "content": explanation_prompt})

                response2   = client.chat.completions.create(model=MODEL, messages=history)
                explanation = response2.choices[0].message.content
                history.append({"role": "assistant", "content": explanation})

                print(f"\nAgent: {explanation}")

            except Exception as e:
                print(f"\n[Local Error] {e}")
                print(f"Raw reply: {agent_reply}")

        # Branch 2: agent is asking for missing information
        else:
            print(f"\nAgent: {agent_reply}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        model, feature_cols = load_phase1_artifacts()
    except FileNotFoundError:
        print("[Error] model.pkl / feature_cols.pkl not found.")
        print("        Run phase1_pipeline.py first.")
        exit(1)

    client = AzureOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        api_key=API_KEY,
        api_version=API_VERSION,
    )

    run_conversation(client, model, feature_cols)