import os
import ast
import sys
import yaml
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent


class DebateManager:
    def __init__(self, debate_group, agents, topic, question, debate_rounds, debate_iterations, taxonomy, debate_structures): #is_structured, give_full_tree):
        # agent config
        self.debate_group = debate_group
        self.agents = agents

        self.taxonomy = taxonomy
        
        # debate config
        self.debate_topic = topic
        self.debate_question = question
        self.num_debate_rounds = debate_rounds
        self.debate_iterations = debate_iterations
        self.debate_structures = debate_structures
        
        # info for file saving
        self.ordered_conversation_history = []  # [{"agent: response"}, ...]
        self.round_num_counts = {"neutral": 0, "democrat": 0, "republican": 0}

        self.debate_start_time = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.data_for_evaluation = {  # used for evaluation
            "topic": self.debate_topic,
            "debate_question":self.debate_question,
            "timestamp":  self.debate_start_time,
            "agents": [ {
                "agent_id": index,
                "name": agent.name,
                "party": agent.party,
                "persona": agent.persona_prompt,
            } for index, agent in enumerate(self.agents)],
            "neutral": {},
            "republican": {},
            "democrat": {}
        }

        self.debate_prompt = self._generate_debate_prompt()


    def _generate_debate_prompt(self):
        prompt = f"""
        You are participating in a political debate on the question: "{self.debate_question}". Keep your reply shorter than 50 words.
        """
        return prompt


    def _clear_data(self):
        self.ordered_conversation_history = []
        self.round_num_counts = {"neutral": 0, "democrat": 0, "republican": 0}

        self.debate_start_time = datetime.now().strftime("%d%m%Y_%H%M%S")
        self.data_for_evaluation = {  # used for evaluation
            "topic": self.debate_topic,
            "debate_question":self.debate_question,
            "timestamp":  self.debate_start_time,
            "agents": [ {
                "agent_id": index,
                "name": agent.name,
                "party": agent.party,
                "persona": agent.persona_prompt,
            } for index, agent in enumerate(self.agents)],
            "neutral": {},
            "republican": {},
            "democrat": {}
        }


    def start(self):

        if self.taxonomy != None:
            # for _ in range(self.debate_iterations):
            for debate_structure in self.debate_structures:
                if debate_structure[0] == "taxonomic":
                    if debate_structure[1] == "full_tree":
                        print("tax, full tree")
                        self._start_taxonomic_debate_with_full_tree()
                        pass
                    elif debate_structure[1] == "traversal":
                        self._start_taxonomic_debate_via_traversal()
                elif debate_structure[0] == "non_taxonomic":
                    self._start_non_taxonomic_debate()

                self._save_evaluation_data("_".join(debate_structure))  # save debates for evaluation
                self._clear_data()


    def _print_response(self, agent_name, response):
        print(f"{agent_name} > {response}")

    
    def _debate_round(self, agent, debate_phase_prompt):
        conversation = "\n".join(self.ordered_conversation_history)
        raw_response = agent.respond(debate_phase_prompt, conversation, inst_prompt=self.debate_prompt)
        # remove escape characters and surrounding quotes safely
        try:
            response = ast.literal_eval(raw_response) if isinstance(raw_response, str) else raw_response
        except Exception:
            response = raw_response.strip('"').replace('\\"', '"')

        self._print_response(agent.name, response)

        self.ordered_conversation_history.append(f"{agent.name}: {response}")

        round_num = self.round_num_counts[agent.identifier]
        self.data_for_evaluation[agent.identifier][f"round_{round_num}"] = response
        self.round_num_counts[agent.identifier] += 1

        return response


    def _start_non_taxonomic_debate(self):
        
        for agent in self.agents:
            self._debate_round(agent, "Present your opening statement.")

        for _ in range(1, self.num_debate_rounds - 2): 
            for agent in self.agents:
                self._debate_round(agent, "Complete your next reply.")

        for agent in self.agents:
            self._debate_round(agent, "Present your closing statement.")


    def _start_taxonomic_debate_with_full_tree(self):
        for agent in self.agents:
            # self._debate_round(agent, f"Present your opening statement using the taxonomy for the question '{self.debate_question}': '{self.taxonomy}'")
            self._debate_round(agent, f"Present your opening statement using the taxonomy: '{self.taxonomy}'")

        for _ in range(1, self.num_debate_rounds - 2): 
            for agent in self.agents:
                self._debate_round(agent, f"Complete your next reply using the taxonomy: '{self.taxonomy}")

        for agent in self.agents:
            self._debate_round(agent, f"Present your closing statement using the taxonomy: '{self.taxonomy}")


    def _start_taxonomic_debate_via_traversal(self):

        for agent in self.agents:
            self._debate_round(agent, "Present your opening statement for the debate.")

        _, discussion_points = next(iter(self.taxonomy.items()))
        discussion_points = dict(list(discussion_points.items())[:1])
        for discussion_point, arguments in discussion_points.items():
            print("\n" + "="*60)
            print(f"\nDiscussion Point: {discussion_point}\n")

            print(f"Arguments: {arguments}\n")
            print("="*60 + "\n")

            # for argument, counterarguments in arguments.items():
            #     print(f"Starting sub-debate for argument: '{argument}'\n")
            #     print("="*60 + "\n")

                # for agent in self.agents:
                #     self._debate_round(agent, f"Present your opening statement for the argument: '{argument}'.")

            argument_rounds = 3
            for _ in range(argument_rounds): 
                for agent in self.agents:
                    self._debate_round(agent, f"Complete your next reply for the debate using the taxonomy for the discussion point '{discussion_point}': '{arguments}'.")

        for agent in self.agents:
            self._debate_round(agent, "Present your closing statement for the debate.")
            

    def get_relative_path(self, filename, folder="debate"):
        # to enable running from root folder or debate subfolder
        if os.path.basename(os.getcwd()) == folder:
            return filename
        return os.path.join(folder, filename)


    def get_relative_path(self, filename, folder="debate"):
        # to enable running from root folder or debate subfolder
        if os.path.basename(os.getcwd()) == folder:
            return filename
        return os.path.join(folder, filename)


    def _save_evaluation_data(self, debate_structure):
        save_folder = f"data/eval_data/{self.debate_group}/{debate_structure}/{self.debate_topic.replace(' ', '_') if self.debate_topic else self.debate_question[:-1].replace(' ', '_')}"
        os.makedirs(save_folder, exist_ok=True)
        filename = f"{save_folder}/transcript_{self.debate_start_time}.json"
    
        with open(filename, "w") as file:
            json.dump(self.data_for_evaluation, file, indent=4)

        print(f"Evaluation data saved:\n- {filename}")


def load_latest_taxonomy(topic):
    topic = topic.replace(" ", "_").lower()
    folder = f"data/taxonomy/{topic}"

    if not os.path.exists(folder):
        print(f"No saved taxonomies found for topic: {topic}\n")
        return None

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        print(f"No taxonomy files found in {folder}\n")
        return None

    files.sort(reverse=True)
    latest_file = files[0]

    file_path = os.path.join(folder, latest_file)

    with open(file_path, "r", encoding="utf-8") as file:
        taxonomy = json.load(file)

    print(f"Loaded latest taxonomy from: {file_path}\n")
    return taxonomy


if __name__ == "__main__":

    # load debate config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debate_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if len(sys.argv) > 1:
        topics = sys.argv[1:]
    else:
        topics = config["baseline_debate_topics"]

    print(f"Debate topics selected: {topics}\n")
    print(f"Debate structures selected: {config['debate_structures']}\n")

    debate_questions = config["baseline_debate_questions"]

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
    print("\n" + "="*60)
    print(f"Agent personas")
    print("="*60 + "\n")
    for agent in debate_agents:
        agent._generate_persona_prompt()
        print(agent.persona_prompt + "\n")

    # generate debate
    for topic, question in zip(topics, debate_questions):
        print("\n" + "="*60)
        print(f"Starting debate for {topic}")
        print("="*60 + "\n")
        # load latest taxonomy for this topic
        taxonomy = load_latest_taxonomy(topic)

        debate_manager = DebateManager(
            debate_group=config["debate_group"],
            agents=debate_agents, 
            topic=topic, 
            question=question, 
            taxonomy=taxonomy, 
            debate_rounds=config["debate_rounds"], 
            debate_iterations=config["debate_iterations"],
            debate_structures = config["debate_structures"]
            # is_structured = config["is_structured"],
            # give_full_tree=config["give_full_tree"]
        )
        debate_manager.start()
