import sys
import os
import yaml
import re
import ast
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent


class TaxonomyGenerator:
    def __init__(self, agents, topic, rounds):
        self.topic = topic
        self.agents = agents
        self.rounds = rounds

        self.ordered_conversation_history = []  # [{"agent: response"}, ...]
        self.conversation_for_taxonomy = []  # [{"agent": agent, "response": response}, ...]


    def start(self, num_debates):
        # for _ in range(num_debates):  # TODO: save debate data (transcript) + final taxonomy; clear data before every run 
        self._start_taxonomy_debate()

        for entry in reversed(self.conversation_for_taxonomy):
            # get the last agreed taxonomy
            if "Taxonomy =" in entry["response"]:
                taxonomy_str = entry["response"]
                break

        taxonomy = self._parse_taxonomy(taxonomy_str)

        self.save_taxonomy(taxonomy)

        return taxonomy


    def _clear_data(self):
        self.ordered_conversation_history = []
        self.conversation_for_taxonomy = []


    def _parse_taxonomy(self, taxonomy_str):
        match = re.search(r"Taxonomy = (\{.*\})", taxonomy_str, re.DOTALL)

        if match:
            taxonomy_str = match.group(1)  # extract only dictionary part
            taxonomy_dict = ast.literal_eval(taxonomy_str)  # convert string to dictionary
            print(taxonomy_dict)

            return taxonomy_dict
        else:
            print("No taxonomy found.")


    def _start_taxonomy_debate(self):
        
        for agent in self.agents:
            self._debate_round(agent, "Present your opening taxonomy for this topic.")
        
        for _ in range(self.rounds - 1): 
            for agent in self.agents:
                self._debate_round(agent, "Update the taxonomy based on the debate so far.")

        # for agent in self.agents:
        #     self._debate_round(agent, "Present your closing taxonomy for this topic, coming to a consensus with the other agent(s).")


    def _debate_round(self, agent, debate_phase_prompt=None):
        conversation = "\n".join(self.ordered_conversation_history)
        response = agent.respond(debate_phase_prompt, conversation)

        self._print_response(agent.name, response)

        self.ordered_conversation_history.append(f"{agent.name}: {response}")
        self.conversation_for_taxonomy.append({"agent": agent, "response": response})


    def _print_response(self, agent_name, response):
        print(f"{agent_name} > {response}")


    def save_taxonomy(self, taxonomy):
        topic = self.topic.replace(" ", "_").lower()
        save_folder = f"data/taxonomy/{topic}"
        os.makedirs(save_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"{save_folder}/{topic}_{timestamp}.json"

        print(type(taxonomy))

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(taxonomy, file, indent=4, ensure_ascii=False)

        print(f"Taxonomy saved to {filename}")



if __name__ == "__main__":
    model1 = "llama3.2:3b"
    model2 = "mistral:7b"
    debate_topic = "climate change"

    # TODO: current prompt is geared for political debates
    taxonomy_prompt = f"""
    You are participating in a structured, political debate on the topic: "{debate_topic}". Your goal is to construct a taxonomy for the political debate using a tree structure where:
    1. The root node represents the overarching debate topic.
    2. The first level consists of major perspectives for the debate topic.
    3. The second level breaks down each perspective into subcategories supporting and opposing the perspective.
    4. Additional levels refine arguments further if needed.
    Output the taxonomy as a dictionary like in the example.
    Provide an explanation of your categorisation in less than 50 words.
    """ \
    """
    \nUse the format in the example below:
    Taxonomy = {
        "Debate Topic": {
            "Perspective 1": {
                "Argument A": {
                    "Supporting Point 1": {},
                    "Supporting Point 2": {}
                },
                "Argument B": {}
            },
            "Perspective 2": {
                "Argument C": {
                    "Counterargument X": {},
                    "Counterargument Y": {}
                }
            }
        }
    }

    Explanation: The taxonomy organizes the debate into two primary perspectives: Perspective 1 and Perspective 2. Each perspective contains key arguments, which are further divided into supporting points or counterarguments. This hierarchical structure helps in understanding the debate by breaking down complex arguments into smaller, manageable parts.
    """


    # create agents
    agent1 = DebateAgent(name="Maven", model=model1, inst_prompt=taxonomy_prompt, stance="proponent")
    agent2 = DebateAgent(name="Ray", model=model2, inst_prompt=taxonomy_prompt, stance="opponent")
    agents = [agent1, agent2]

    # generate taxonomy
    taxonomy_gen = TaxonomyGenerator(agents=agents, topic=debate_topic, rounds=3)
    taxonomy = taxonomy_gen.start(num_debates=1)


    

    # two sides = proposition (proponent) & opposition (skeptic) -- British Parliamentary style

    # calculate simple metrics that show improved debate quality
