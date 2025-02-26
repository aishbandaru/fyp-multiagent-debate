import sys
import os
import yaml
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent
# from debate.DebateManager import DebateManager


class DebateManager:
    def __init__(self, agents, topic, rounds, taxonomy):
        self.topic = topic
        self.agents = agents
        self.rounds = rounds
        self.taxonomy = taxonomy

        self.ordered_conversation_history = []  # [{"agent: response"}, ...]
        self.conversation_for_transcript = []  # [{"agent": agent, "response": response}, ...]

    
    def start(self, num_debates):
        # for _ in range(num_debates):  # TODO: save debate data (transcript) + final taxonomy; clear data before every run 
        self._start_taxonomic_debate()

        self._save_debate_transcription()


        # clear data
        # self._start_simple_debate()

        # save debates for evaluation!
        
        # evaluate debates (come up with metrics on repetition, coherence etc.)


    def _print_response(self, agent_name, response):
        print(f"{agent_name} > {response}")

    
    def _debate_round(self, agent, debate_phase_prompt=None):
        conversation = "\n".join(self.ordered_conversation_history)
        response = agent.respond(debate_phase_prompt, conversation)

        self._print_response(agent.name, response)

        self.ordered_conversation_history.append(f"{agent.name}: {response}")
        self.conversation_for_transcript.append({"agent": agent, "response": response})

        return response


    def _start_simple_debate(self):
        
        for agent in self.agents:
            # self.debate_round(agent, "Present your opening opinions on the topic. Do not rebut the other agent. Do not disagree with them.")
            self.debate_round(agent, "Present your opening stance towards the debate motion. Keep your reply shorter than 50 words.")

        for _ in range(1, self.rounds - 1): 
            for agent in self.agents:
                # self.debate_round("Please rebut the other agent's opinions and continue to argue your own.", agent)
                self.debate_round(agent, "Complete your next reply. Keep your reply shorter than 50 words.")  # NOTE: This takes the prompt used in baseline paper



    def _start_taxonomic_debate(self):

        topic_name, perspectives = next(iter(self.taxonomy.items()))
        
        for perspective, arguments in perspectives.items():
            print(f"📌 **Perspective:** {perspective}\n")

            print(f"📌 **Arguments:** {arguments}\n")

            # (TODO: give the agents all the arguments they will debate on at the beginning?)

            for argument, counterarguments in arguments.items():
                print(f"\n--- Debating Argument: '{argument}' ---")

                for agent in self.agents:
                    self._debate_round(agent, f"Present your opening stance towards the argument: '{argument}'. Keep your reply shorter than 50 words.")

                for _ in range(self.rounds - 1): 
                    for agent in self.agents:
                        self._debate_round(agent, "Complete your next reply based on the debate so far. Keep your reply shorter than 50 words.")

                # print(f"{self.agents[0].name} presents opening stance on: '{argument}'")
                # debate_phase_prompt = f"Present your opening stance towards the argument: '{argument}'. Keep your reply shorter than 50 words."
                # self._debate_round(self.agents[0], debate_phase_prompt)

                # # Agent 2 counters Agent 1's argument
                # print(f"{self.agents[1].name} counters: '{argument}'")
                # debate_phase_prompt = f"Present your counterargument to: '{argument}'. Keep your reply shorter than 50 words."
                # self._debate_round(self.agents[1], debate_phase_prompt)

                print("\n---\n")

        print("\n🎤 Debate Concludes 🎤")

        # print(self.ordered_conversation_history)
            

    def get_relative_path(self, filename, folder="debate"):
        # to enable running from root folder or debate subfolder
        if os.path.basename(os.getcwd()) == folder:
            return filename
        return os.path.join(folder, filename)


    def _save_debate_transcription(self):
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

        save_folder = self.get_relative_path(f"data/debate_transcripts/{self.debate_group}/{self.debate_structure}/{self.topic.replace(' ', '_')}")
        os.makedirs(save_folder, exist_ok=True)

        # TXT
        text_filename = f'{save_folder}/transcript_{timestamp}.txt'
        with open(text_filename, 'w') as f:
            for round in self.conversation_for_transcription:
                f.write(f'{round["agent"].label} > {round["response"]} \n\n')

        # JSON
        json_filename = f'{save_folder}/transcript_{timestamp}.json'
        json_data = {
            "topic": self.topic,
            "timestamp": timestamp,
            "agents": [ {
                "agent_id": index,
                "name": agent.name,
                "stance": agent.stance, 
            } for index, agent in enumerate(self.agents)],
            "rounds": [
                    {
                        "agent_id": self.agents.index(round["agent"]),
                        "response": round["response"]}
                for round in self.conversation_for_transcription
            ]
        }

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)

        print(f"Transcripts saved:\n- {text_filename}\n- {json_filename}")


    

def load_latest_taxonomy(topic):
    topic = topic.replace(" ", "_").lower()
    folder = f"data/taxonomy/{topic}"

    if not os.path.exists(folder):
        print(f"No saved taxonomies found for topic: {topic}")
        return None

    files = [f for f in os.listdir(folder) if f.endswith(".json")]
    if not files:
        print(f"No taxonomy files found in {folder}")
        return None

    files.sort(reverse=True)
    latest_file = files[0]

    file_path = os.path.join(folder, latest_file)

    with open(file_path, "r", encoding="utf-8") as file:
        taxonomy = json.load(file)

    print(f"Loaded latest taxonomy from: {file_path}")
    return taxonomy


if __name__ == "__main__":
    model1 = "llama3.2:3b"
    model2 = "mistral:7b"
    debate_topic = "climate change"

    debate_prompt = ""  # TODO

    # create agents
    agent1 = DebateAgent(name="Maven", model=model1, inst_prompt=debate_prompt, stance="proponent")
    agent2 = DebateAgent(name="Ray", model=model2, inst_prompt=debate_prompt, stance="opponent")
    agents = [agent1, agent2]


    # load latest taxonomy for this topic
    taxonomy = load_latest_taxonomy(debate_topic)

    # debate taxonomy
    debate_manager = DebateManager(agents=agents, topic=debate_topic, rounds=1, taxonomy=taxonomy)
    debate_manager.start(num_debates=1)




    # 3 rounds of debate to agree on an ontology
    
