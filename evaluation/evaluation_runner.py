import os
import sys
import yaml
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.DebateEvaluator import DebateEvaluator


if __name__ == "__main__":
    # load evaluation config from YAML
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    if len(sys.argv) > 1:
        debate_topics = sys.argv[1:]
    else:
        debate_topics = config["debate_topics"]

    model = config["model"]
    debate_group = config["debate_group"]  # "neutral_republican", "neutral_democrat", or "neutral_republican_democrat"
    debate_structures = config["debate_structures"]

    base_transcripts_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "debate_data", debate_group)

    print(f"Selected debate group: {debate_group}")
    print(f"Selected debate topics: {debate_topics}\n")
    start_time = datetime.now()

    transcripts = {}

    for structure in debate_structures:
        structure_path = os.path.join(base_transcripts_path, structure)
        for topic in debate_topics:
            topic_path = os.path.join(structure_path, topic)
            if os.path.exists(topic_path) and os.listdir(topic_path):
                transcripts_key = f"{structure}/{topic}"
                transcripts[transcripts_key] = sorted(
                    os.listdir(topic_path),
                    key=lambda x: os.path.getmtime(os.path.join(topic_path, x)),
                    reverse=True
                )[:config["num_debate_iterations"]]

    debate_evaluator = DebateEvaluator(model, debate_group, debate_structures, config["num_debate_rounds"], config["num_debate_iterations"], metrics=config["metrics"], evaluate_again=config["evaluate_again"], debate_topics=debate_topics)

    debate_evaluator.evaluate_debates(base_transcripts_path=base_transcripts_path, debate_transcripts=transcripts)

    print(f"All evaluation has completed for topics: {debate_topics}")
    print(f"That took {(datetime.now() - start_time).total_seconds():.2f} seconds")
