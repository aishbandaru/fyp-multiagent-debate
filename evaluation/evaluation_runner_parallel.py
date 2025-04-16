import os
import sys
import yaml
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.DebateEvaluator import DebateEvaluator

def evaluate_single_transcript(debate_evaluator, transcript_path):
    """Evaluate a single transcript (inner parallel task)"""
    return debate_evaluator.evaluate_transcript(transcript_path)

def evaluate_structure_topic(debate_evaluator, structure_topic, transcript_list, base_transcripts_path):
    """Process all transcripts for one structure-topic (outer parallel task)"""
    all_attitude_scores = {}
    debate_evaluator.structure_topic_path = structure_topic

    if "attitude" in debate_evaluator.metrics:
        for agent in debate_evaluator.debate_group:
            all_attitude_scores[agent] = [[] for _ in range(debate_evaluator.num_iterations)]

    # Use ProcessPoolExecutor for inner parallelism
    with ProcessPoolExecutor() as inner_executor:
        # Prepare tasks for inner parallel processing
        futures = {
            inner_executor.submit(
                evaluate_single_transcript,
                debate_evaluator,
                os.path.join(base_transcripts_path, structure_topic, transcript)
            ): i for i, transcript in enumerate(transcript_list)
        }

        # Process results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                transcript = transcript_list[idx]
                print(f"Evaluating: {structure_topic} - {transcript}\nEvaluation scores: ", result)
                
                if "attitude" in debate_evaluator.metrics:
                    scores = result.get("attitude", None)
                    for agent in debate_evaluator.debate_group:
                        all_attitude_scores[agent][idx] = scores[agent]
            except Exception as e:
                print(f"Error processing {transcript_list[idx]}: {str(e)}")

    if "attitude" in debate_evaluator.metrics:
        debate_evaluator._compute_attitude_metrics(all_attitude_scores)
        topic_name = structure_topic.split("/")[-1]
        debate_evaluator._generate_attitude_box_plot(all_attitude_scores, topic_name)

    print(f"Completed evaluation for {structure_topic}")
    return structure_topic

def evaluate_debates_parallel(debate_evaluator, base_transcripts_path, debate_transcripts):
    """Main parallel evaluation function with two-level parallelism"""
    # Use ProcessPoolExecutor for outer parallelism
    with ProcessPoolExecutor() as outer_executor:
        # Prepare all tasks
        futures = [
            outer_executor.submit(
                evaluate_structure_topic,
                debate_evaluator,
                st,
                tl,
                base_transcripts_path
            ) for st, tl in debate_transcripts.items()
        ]

        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error in structure-topic evaluation: {str(e)}")

if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    
    debate_topics = sys.argv[1:] if len(sys.argv) > 1 else config["debate_topics"]
    model = config["model"]
    debate_group = config["debate_group"]
    debate_structures = config["debate_structures"]

    base_transcripts_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "..", "data", "debate_data", debate_group
    )
    
    print(f"Selected debate group: {debate_group}")
    print(f"Selected debate topics: {debate_topics}\n")
    start_time = datetime.now()

    # Collect transcripts
    transcripts = {}
    for structure in debate_structures:
        structure_path = os.path.join(base_transcripts_path, structure)
        for topic in debate_topics:
            topic_path = os.path.join(structure_path, topic)
            if os.path.exists(topic_path) and os.listdir(topic_path):
                key = f"{structure}/{topic}"
                transcripts[key] = sorted(
                    os.listdir(topic_path),
                    key=lambda x: os.path.getmtime(os.path.join(topic_path, x)),
                    reverse=True
                )[:config["num_debate_iterations"]]

    # Initialize evaluator
    debate_evaluator = DebateEvaluator(
        model,
        debate_group,
        debate_structures,
        config["num_debate_rounds"],
        config["num_debate_iterations"],
        metrics=config["metrics"],
        evaluate_again=config["evaluate_again"],
        debate_topics=debate_topics
    )

    # Run evaluation
    evaluate_debates_parallel(debate_evaluator, base_transcripts_path, transcripts)

    print(f"\nAll evaluation completed for topics: {debate_topics}")
    print(f"Total time: {(datetime.now() - start_time).total_seconds():.2f} seconds")