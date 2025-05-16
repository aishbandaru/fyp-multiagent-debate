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
        self.num_discussion_points = 4  # number of main key points to generate per debate question
        self.taxonomy_prompt = self._get_taxonomy_prompt()

        self.agents = agents
        self.taxonomy_rounds = taxonomy_rounds
        self.taxonomy_iterations = taxonomy_iterations

        self.ordered_conversation_history = []  # [{"agent: response"}, ...]
        self.conversation_for_taxonomy = []  # [{"agent": agent, "response": response}, ...]


    def _get_taxonomy_prompt(self):
        taxonomy_prompt = f"""
        You are an expert in structuring political debates. Generate a taxonomy for the debate topic '{self.debate_question}' in JSON format. Collaborate with the other experts. This taxonomy includes:
        1. A root node for the debate topic
        2. A first level with at most {self.num_discussion_points} arguments.
        3. A second level with at most 2 subpoints per argument.
        4. A third level with at most 3 details per subpoint.
        5. No further levels.

        It is used for a political debate generation system. Explain your taxonomy in less than 50 words.
        """ \
        """
        \nUse the following format, using 10 words or less for each item:
        {
            "topic": "Topic question",
            "arguments": [
                {
                "point": "First main argument",
                "subpoints": [
                    {
                    "point": "Subargument A",
                    "details": [
                        "Supporting detail 1",
                        "Supporting detail 2",
                        "Supporting detail 3"
                    ]
                    }
                ]
                },
                ... (3 more points)
            ]
        }
        """
        return taxonomy_prompt


    def start(self):
        for _ in range(self.taxonomy_iterations):
            self._start_taxonomy_debate()

            taxonomy_str = None

            # iterate in reverse to find the last entry from the neutral agent to use as the taxonomy
            print(f"\n[DEBUG] Total entries in conversation_for_taxonomy: {len(self.conversation_for_taxonomy)}")

            for entry in reversed(self.conversation_for_taxonomy):
                if entry.get("agent") == "neutral":
                    response = entry.get("response", "").strip()
                    if "```json" in response:
                        # start of JSON block
                        start_idx = response.find("```json") + len("```json")
                        # end of JSON block (looking for the first closing ``` after the JSON block)
                        end_idx = response.find("```", start_idx)

                        if start_idx != -1 and end_idx != -1:
                            # extract the JSON taxonomy content
                            taxonomy_str = response[start_idx:end_idx].strip()
                            break

            if not taxonomy_str:
                print("[ERROR] No taxonomy response found from neutral agent.")

            if taxonomy_str:
                taxonomy = self._parse_taxonomy(taxonomy_str)
                self.save_taxonomy(taxonomy)
            else:
                print("[ERROR] Could not extract valid taxonomy.")

            self._clear_data()


    def _clear_data(self):
        self.ordered_conversation_history = []
        self.conversation_for_taxonomy = []


    def _parse_taxonomy(self, taxonomy_str):
        # extract JSON code block content
        match = re.search(r"```json\s*(\{.*?\})\s*```", taxonomy_str, re.DOTALL)

        if not match:
            # fallback to matching JSON in the plain string, with or without prefix
            match = re.search(r"(?:Taxonomy\s*=\s*)?(\{.*\})", taxonomy_str, re.DOTALL)

        if match:
            json_str = match.group(1)

            # custom function to fix subarguments: ensure they are either strings or empty objects
            def fix_subarguments(data):
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, str):
                            continue
                        elif isinstance(value, dict) and not value:
                            continue
                        else:
                            fix_subarguments(value)
                return data

            try:
                print("\nTaxonomy String before parsing:", taxonomy_str)
                taxonomy_dict = json.loads(json_str)

                # resolve subarguments after parsing the JSON
                taxonomy_dict = fix_subarguments(taxonomy_dict)
                print("Parsed taxonomy successfully.")
                return taxonomy_dict

            except Exception as e:
                print("Error parsing taxonomy JSON:", e)
                raise
        else:
            print("No valid taxonomy JSON found.")
            return None


    def _start_taxonomy_debate(self):
        for agent in self.agents:
            self._debate_round(agent, "Present your initial taxonomy of the debate topic.")

        for _ in range(1, self.taxonomy_rounds - 1): 
            for agent in self.agents:
                self._debate_round(agent, "Update the taxonomy based on previous contributions. You may refine, merge, or extend existing points.")

        for agent in self.agents:
            self._debate_round(agent, "Present your final taxonomy of the debate topic based on previous contributions.")


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
        save_folder = f"data/new_taxonomy/multiagent/{topic}/temp_{agent.temperature}"  # NOTE: Change save_folder depending on neutral_only vs multiagent generation
        os.makedirs(save_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

        filename = f"data/new_taxonomy/multiagent/{topic}/temp_{agent.temperature}_best.json" 

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
    start_time = datetime.now()


    for temperature in [0.2, 0.4, 0.6, 0.8, 1.0]:
        print(f"Starting taxonomy creation for topics: {topics} with agent temperature={temperature}\n")

        # create agents
        debate_agents = []
        for agent_cfg in config["debate_agents"]:
            agent = DebateAgent(
                identifier=agent_cfg["identifier"],
                name=agent_cfg["name"],
                party=agent_cfg["party"],
                leaning=agent_cfg["leaning"],
                model=agent_cfg["model"],
                temperature=temperature
            )
            debate_agents.append(agent)
        
        # generate agent personas and verify them
        print("\n" + "="*60)
        print(f"Agent personas")
        print("="*60 + "\n")
        for agent in debate_agents:
            agent.generate_persona_prompt(use_extended_persona=True)
            print(agent.persona_prompt + "\n")

        # generate taxonomy
        for topic, question in zip(topics, debate_questions):
            print("\n" + "="*120)
            print(f"Starting debate for {topic}: {question}")
            print("="*120 + "\n")
            taxonomy_gen = TaxonomyGenerator(
                agents=debate_agents, 
                debate_topic=topic, 
                debate_question=question, 
                taxonomy_rounds=config["taxonomy_rounds"], 
                taxonomy_iterations=config["taxonomy_iterations"]
            )
            taxonomy_gen.start()

    print(f"All taxonomy generation has completed for topics: {topics}")
    print(f"That took {(datetime.now() - start_time).total_seconds():.2f} seconds")
