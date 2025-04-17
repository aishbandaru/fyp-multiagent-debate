import os
import re
import ast
import sys
import yaml
import json
import multiprocessing
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent
from debate_runner import DebateManager, load_latest_taxonomy

def run_single_debate(topic, question, config, debate_structures, debate_group):
    try:
        taxonomy = load_latest_taxonomy(topic)

        # Create agents
        group_identifiers = debate_group.split("_")
        agents = []
        for agent_cfg in config["debate_agents"]:
            if agent_cfg["identifier"] in group_identifiers:
                agent = DebateAgent(
                    identifier=agent_cfg["identifier"],
                    name=agent_cfg["name"],
                    party=agent_cfg["party"],
                    leaning=agent_cfg["leaning"],
                    model=agent_cfg["model"],
                    temperature=config["temperature"]
                )
                agent.generate_persona_prompt(use_extended_persona=True)
                agents.append(agent)

        # Initialize debate manager
        manager = DebateManager(
            debate_group=debate_group,
            agents=agents,
            topic=topic,
            question=question,
            debate_rounds=config["debate_rounds"],
            debate_iterations=config["debate_iterations"],
            taxonomy=taxonomy,
            debate_structures=debate_structures
        )

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting debate: {topic}")
        manager.start()
    except Exception as e:
        print(f"Error in topic '{topic}': {e}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    # Load debate config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debate_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Get topics to debate
    if len(sys.argv) > 1:
        topics = sys.argv[1:]
    else:
        topics = config["baseline_debate_topics"]

    debate_questions = config["baseline_debate_questions"]
    debate_group = config["debate_group"]
    debate_structures = config["debate_structures"]

    start_time = datetime.now()
    print(f"Running parallel debates for topics: {topics}\n")
    print("Selected debate")

    # Launch parallel debates
    # for topic, question in zip(topics, debate_questions):
    #     run_single_debate(topic, question, config, debate_structures, debate_group)
    processes = []
    for topic, question in zip(topics, debate_questions):
        p = multiprocessing.Process(target=run_single_debate, args=(
            topic, question, config, debate_structures, debate_group
        ))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(f"\nAll debates completed in {datetime.now() - start_time}\n")
