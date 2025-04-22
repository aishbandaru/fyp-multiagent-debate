import os
import re
import ast
import sys
import yaml
import json
from datetime import datetime

# add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent


class TaxonomyGenerator:
    def __init__(self, agents, debate_topic, debate_question, taxonomy_rounds, taxonomy_iterations):
        self.debate_topic = debate_topic
        self.debate_question = debate_question
        self.taxonomy_prompt = self._get_taxonomy_prompt()

        self.agents = agents
        self.taxonomy_rounds = taxonomy_rounds
        self.taxonomy_iterations = taxonomy_iterations
        self.num_discussion_points = 4  # number of main key points to generate per debate question

        self.ordered_conversation_history = []  # [{"agent: response"}, ...]
        self.conversation_for_taxonomy = []  # [{"agent": agent, "response": response}, ...]


    def _get_taxonomy_prompt(self):
        # You are participating in a political debate on the topic "{debate_topic}" for the debate motion: "{debate_motion}".
        taxonomy_prompt = f"""
        You are an expert in structuring political debates. Generate a taxonomy for the debate topic '{self.debate_question}' in JSON format. Collaborate with the other experts. This taxonomy includes:
        1. A root node for the debate topic
        2. A first level with {self.num_discussion_points} major discussion points for the topics
        3. A second level with arguments for each discussion point.
        4. A third level which refines each argument further.

        Explain your taxonomy in less than 50 words.
        """ \
        """
        \nUse the format in the example below:
        Taxonomy = {
            "Debate Topic": {
                "Discussion Point 1": {
                    "Argument A": {
                        "Subargument 1": {},
                        "Subargument 2": {}
                    },
                    "Argument B": {
                        "Subargument 3": {},
                        "Subargument 4": {}
                    }
                },
                "Discussion Point 2": {
                    "Argument C": {
                        "Subargument X": {},
                        "Subargument Y": {}
                    }
                }
            }
        }

        Explanation: The taxonomy organises the debate into two primary discussion points: Discussion Point 1 and Discussion Point 2. Each points contains key arguments, which are further divided into supporting points or counterarguments.
        """
        return taxonomy_prompt


    def start(self):
        # for _ in range(self.taxonomy_iterations):  # TODO: save debate data (transcript) + final taxonomy; clear data before every run 
        self._start_taxonomy_debate()

        taxonomy_str = None

        # iterate in reverse to find the last entry from the neutral agent to use as the taxonomy

        # self.conversation_for_taxonomy.reverse()  # This will modify the list in place
        # print("TESTING SOMETHING", self.conversation_for_taxonomy[0]["agent"])

        for entry in reversed(self.conversation_for_taxonomy):
            if entry["agent"] == "neutral":
                if "Taxonomy =" in entry["response"]:
                    taxonomy_str = entry["response"]
                    break

        # for entry in reversed(self.conversation_for_taxonomy):
        #     # get the last agreed taxonomy
        #     if "Taxonomy =" in entry["response"]:
        #         taxonomy_str = entry["response"]
        #         break

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
            self._debate_round(agent, "Present your initial taxonomy of the debate topic.")

        for _ in range(1, self.taxonomy_rounds - 1): 
            for agent in self.agents:
                self._debate_round(agent, "Update the taxonomy based on previous contributions by other agents. You may refine, merge, or extend existing points.")  # Complete your next reply based on the taxonomy so far

        for agent in self.agents:
            self._debate_round(agent, "Present your final taxonomy of the debate topic.")


    def _debate_round(self, agent, debate_phase_prompt=None):
        conversation = "\n".join(self.ordered_conversation_history)
        response = agent.respond(debate_phase_prompt, conversation, self.taxonomy_prompt)

        self._print_response(agent.name, response)

        self.ordered_conversation_history.append(f"{agent.name}: {response}")
        self.conversation_for_taxonomy.append({"agent": agent.identifier, "response": response})


    def _print_response(self, agent_name, response):
        print(f"{agent_name} > {response}")


    def save_taxonomy(self, taxonomy):
        topic = self.debate_topic.replace(" ", "_").lower()
        save_folder = f"data/taxonomy/{topic}"
        os.makedirs(save_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        filename = f"{save_folder}/{topic}_{timestamp}.json"

        print(type(taxonomy))

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(taxonomy, file, indent=4, ensure_ascii=False)

        print(f"Taxonomy saved to {filename}")



if __name__ == "__main__":

    # load debate config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debate_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if len(sys.argv) > 1:
        topics = sys.argv[1:]
    else:
        topics = config["baseline_debate_topics"]
    
    debate_questions = config["baseline_debate_questions"]

    print(f"Starting taxonomy creation for topics: {topics}\n")
    start_time = datetime.now()

    # create agents
    debate_agents = []
    for agent_cfg in config["debate_agents"]:
        agent = DebateAgent(
            identifier=agent_cfg["identifier"],
            name=agent_cfg["name"],
            party=agent_cfg["party"],
            leaning=agent_cfg["leaning"],
            model=agent_cfg["model"],
            temperature=config["temperature"]
        )
        debate_agents.append(agent)
    
    # generate agent personas and verify them
    for agent in debate_agents:
        agent.generate_persona_prompt(use_extended_persona=True)
        print(agent.persona_prompt + "\n")

    # generate taxonomy
    for topic, question in zip(topics, debate_questions):
        taxonomy_gen = TaxonomyGenerator(
            agents=debate_agents, 
            debate_topic=topic, 
            debate_question=question, 
            taxonomy_rounds=config["taxonomy_rounds"], 
            taxonomy_iterations=config["taxonomy_iterations"]
        )
        taxonomy = taxonomy_gen.start()

    print(f"All taxonomy generation has completed for topics: {topics}")
    print(f"That took {(datetime.now() - start_time).total_seconds():.2f} seconds")
