import sys
import os
import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent
from debate.DebateManager import DebateManager

if __name__ == "__main__":
    model = "llama3.2:3b"

    debate_topics = {
        "gun_violence": "Should gun control laws be stricter in the United States?",
    }

    # Prompt agent
    agent1 = DebateAgent(name="NAME", model=model, prompt="PROMPT")


    # 3 rounds of debate to agree on an ontology
    
