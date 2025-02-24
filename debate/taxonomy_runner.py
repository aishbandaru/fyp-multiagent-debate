import sys
import os
import yaml

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent

def start_taxonomy_debate(agents, debate_topic):
    pass

    # structured debate w/ opening, middle and closing rounds

    # if agent.name == agents[-1].name
    # return string with final taxonomy (what the last agent says in closing statement)


def construct_taxonomy(agents, debate_topic)

    taxonomy_str = start_taxonomy_debate(agents, debate_topic)

    taxonomy = parse_taxonomy()  # construct taxonomy tree

    return taxonomy



if __name__ == "__main__":
    model = "llama3.2:3b"
    debate_topic = "climate change"

    taxonomy_prompt = ""

    # create agents
    agent1 = DebateAgent(name="Alex", model=model, inst_prompt=taxonomy_prompt)
    agent2 = DebateAgent(name="Aven", model=model, inst_prompt=taxonomy_prompt)
    agents = [agent1, agent2]

    # generate taxonomy
    taxonomy = construct_taxonomy(agents, debate_topic)


    # save taxonomy
