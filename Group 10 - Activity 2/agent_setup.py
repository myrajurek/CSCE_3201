from azure.ai.projects import AIProjectClient
from azure.core.credentials import AzureKeyCredential

# -------------------------------------------------------------------
# CONFIG - YOUR AZURE VALUES
# -------------------------------------------------------------------

PROJECT_ENDPOINT = "https://metacritic-agent-phase2-resource.services.ai.azure.com/api/projects/Metacritic_Agent_Phase2"
API_KEY = "u6PAcHqxM8O5DfaGxEgRjTPA7aGdATn4WOH0psAQy0PBRtfu3Pk4JQQJ99CDACHYHv6XJ3w3AAAAACOGTyKH" 

# IDENTIFIERS FROM YOUR YAML/PORTAL
AGENT_NAME = "metacritic-score-predictor"
AGENT_VERSION = "2"

# -------------------------------------------------------------------
# VERIFICATION & SAVE
# -------------------------------------------------------------------

def verify_and_save():
    """
    Connects to Azure to verify the project exists, 
    then saves the agent identifiers for the pipeline.
    """
    try:
        # Create client to verify credentials
        client = AIProjectClient(
            endpoint=PROJECT_ENDPOINT,
            credential=AzureKeyCredential(API_KEY)
        )
        print("[Azure] Connected successfully to the project")

        # Save identifiers to the text file
        # We use a 'name:version' format so agent_pipeline.py can parse it easily
        with open("agent_id.txt", "w") as f:
            f.write(f"{AGENT_NAME}:{AGENT_VERSION}")
        
        print(f"\n[Success] Agent Identifiers Saved!")
        print(f"Name: {AGENT_NAME}")
        print(f"Version: {AGENT_VERSION}")
        print("-" * 40)
        print("You are now ready to run agent_pipeline.py")

    except Exception as e:
        print(f"[Error] Could not connect to Azure: {e}")

if __name__ == "__main__":
    verify_and_save()