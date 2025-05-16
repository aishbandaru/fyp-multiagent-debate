import os
import re
import sys
import csv
import json
import yaml
import ollama
from openai import OpenAI
from datetime import datetime


client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TaxonomyEvaluator:
    def __init__(self, root_dir="data/new_taxonomy", mode="neutral_only", model="gpt-4.1-2025-04-14"):
        self.root_dir = root_dir
        self.mode = mode  # either 'neutral_only' or 'multiagent'
        self.model = model
        self.temperatures = [0.2, 0.4, 0.6, 0.8, 1.0]
        self.best_paths_per_topic = {}
        self.across_mode_comparisons = {}


    def load_taxonomies(self, topic, find_best=False):
        taxonomies = []
        topic_path = os.path.join(self.root_dir, self.mode, topic.replace(" ", "_"))

        for temp in self.temperatures:
            if find_best:
                file_path = os.path.join(topic_path, f"temp_{temp}_best.json")
                if os.path.exists(file_path):
                    with open(file_path) as f:
                        taxonomy = json.load(f)
                        taxonomies.append((temp, file_path, taxonomy))
            else:
                temp_dir = os.path.join(topic_path, f"temp_{temp}")
                if not os.path.exists(temp_dir):
                    continue
                for file_name in os.listdir(temp_dir):
                    if file_name.endswith(".json"):
                        file_path = os.path.join(temp_dir, file_name)
                        with open(file_path) as f:
                            taxonomy = json.load(f)
                            taxonomies.append((temp, file_path, taxonomy))

        return taxonomies



    def format_for_prompt(self, taxonomy):
        def format_arg(arg, level=0):
            indent = "  " * level
            lines = [f"{indent}- {arg['point']}"]
            for sub in arg.get("subpoints", []):
                lines.append(f"{indent}  - {sub['point']}")
                for detail in sub.get("details", []):
                    lines.append(f"{indent}    - {detail}")
            return lines

        lines = [f"Topic: {taxonomy['topic']}"]
        for arg in taxonomy.get("arguments", []):
            lines += format_arg(arg)
        return "\n".join(lines)

    def evaluate_batch(self, topic, taxonomies):
        results = {}
        topic_dir = os.path.join(self.root_dir, self.mode, topic.replace(" ", "_"))

        for i, group in enumerate([taxonomies[i:i+3] for i in range(0, len(taxonomies), 3)]):
            if len(group) < 2:
                continue

            prompt = f"Given the debate topic, which taxonomy is better? Answer “Taxonomy 1”, “Taxonomy 2” or “Taxonomy 3”. Explain the reason.\n\nDebate topic: {group[0][2]['topic']}\n"
            for idx, (_, _, taxonomy) in enumerate(group):
                prompt += f"\nTaxonomy {idx + 1}:\n{self.format_for_prompt(taxonomy)}\n"

            print(f"Evaluating batch of {len(group)} taxonomies for topic '{topic}'\n")

            # content = ollama.generate(options={"temperature":0.0}, model="mistral:7b", prompt=prompt)["response"]

            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )

            # content = response.choices[0].message.content
            print(f"LLM Response: {content}\n")

            match = re.search(r"Taxonomy (\d)", content)
            if match:
                chosen_idx = int(match.group(1)) - 1
                if 0 <= chosen_idx < len(group):
                    temp, old_path, taxonomy = group[chosen_idx]
                    best_filename = f"temp_{temp}_best.json"
                    best_path = os.path.join(topic_dir, best_filename)
                    with open(best_path, "w") as f:
                        json.dump(taxonomy, f, indent=2)
                    results[temp] = best_path

        self.best_paths_per_topic[topic] = results


    def evaluate_taxonomy(self, taxonomy):
        prompt = f"""
        Evaluate the quality of the taxonomy for the debate topic ‘{taxonomy['topic']}’. DO NOT EXPLAIN YOUR REASONING.
        Rate each taxonomy on four dimensions:
        1. Completeness: Does the taxonomy cover all important aspects of the topic?
        2. Conciseness: Is the taxonomy free from irrelevant or redundant elements?
        3. Clarity: Are the categories well-labelled and understandable?
        4. Perspective Coverage: Does the taxonomy reflect different political viewpoints relevant to the debate topic?
        You should rate on a scale from 1 (worst) to 10 (best).

        Taxonomy: {self.format_for_prompt(taxonomy)}
        Completeness Score:
        Conciseness Score:
        Clarity Score:
        Perspective Coverage Score:
        """
        
        scores = {
            "Completeness": [],
            "Conciseness": [],
            "Clarity": [],
            "Perspective Coverage": []
        }

        # get 3 evaluations for each taxonomy dimension
        for _ in range(3):
            # content = ollama.generate(options={"temperature":0.0}, model="mistral:7b", prompt=prompt)["response"]
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            content = response.choices[0].message.content

            score_match = re.findall(r"([A-Za-z ]+Score):\s*(\d+(\.\d+)?)", content)  # General match for all scores    

            if score_match:
                for dimension_score, score, _ in score_match:
                    dimension = dimension_score.strip().replace(" Score", "")  # Remove extra spaces and ' Score'
                    if dimension in scores:
                        scores[dimension].append(float(score))  # Ensure we store it as a float

        # compute the average for each dimension
        avg_scores = {dim: sum(scores[dim]) / len(scores[dim]) for dim in scores}
        print(f"Average Scores for topic '{taxonomy['topic']}': {avg_scores}\n")
        return avg_scores


    def evaluate_cross_mode(self):
        cross_results = {}
        topics = os.listdir(os.path.join(self.root_dir, "neutral_only"))
        
        for topic in topics:
            for temp in self.temperatures:
                n_path = os.path.join(self.root_dir, "neutral_only", topic.replace(" ", "_"), f"temp_{temp}_best.json")
                m_path = os.path.join(self.root_dir, "multiagent", topic.replace(" ", "_"), f"temp_{temp}_best.json")
                if not os.path.exists(n_path) or not os.path.exists(m_path):
                    continue

                with open(n_path) as f1, open(m_path) as f2:
                    n_tax = json.load(f1)
                    m_tax = json.load(f2)

                # get scores for both taxonomies
                n_scores = self.evaluate_taxonomy(n_tax)
                m_scores = self.evaluate_taxonomy(m_tax)

                # compare the averaged scores for four dimensions
                n_avg = sum(n_scores.values()) / 4
                m_avg = sum(m_scores.values()) / 4

                winner = "neutral_only" if n_avg > m_avg else "multiagent"
                cross_results[(topic, temp)] = {
                    "winner": winner,
                    "neutral_path": n_path,
                    "multiagent_path": m_path,
                    "n_scores": n_scores,
                    "m_scores": m_scores
                }

        self.across_mode_comparisons = cross_results


    def save_summary_csv(self, out_path="taxonomy_evaluation_summary.csv"):
        rows = []

        # record best taxonomy paths for multiagent and neutral-only taxonomies with results
        for topic, temp_results in self.best_paths_per_topic.items():
            for temp, best_path in temp_results.items():
                rows.append({
                    "topic": topic,
                    "temperature": temp,
                    "mode": self.mode,
                    "best_taxonomy_path": best_path,
                    "comparison_type": "within-mode",
                    "neutral_scores": "",
                    "multiagent_scores": "",
                    "winning_mode": ""
                })

        # add divider row before cross-mode results
        rows.append({
            "topic": "=== CROSS-MODE COMPARISON RESULTS ===",
            "temperature": "", "mode": "", "best_taxonomy_path": "",
            "comparison_type": "", "neutral_scores": "", "multiagent_scores": "", "winning_mode": ""
        })

        # record vross-mode comparisons with full score info
        for (topic, temp), result in self.across_mode_comparisons.items():
            rows.append({
                "topic": topic,
                "temperature": temp,
                "mode": "",
                "best_taxonomy_path": "",
                "comparison_type": "cross-mode",
                "neutral_scores": result["n_scores"],
                "multiagent_scores": result["m_scores"],
                "winning_mode": result["winner"]
            })

        # add divider row before best taxonomy overall results
        rows.append({
            "topic": "=== BEST TAXONOMY OVERALL (BY AVG SCORE) ===",
            "temperature": "", "mode": "", "best_taxonomy_path": "",
            "comparison_type": "", "neutral_scores": "", "multiagent_scores": "", "winning_mode": ""
        })

        # record highest scoring taxonomy overall per topic
        for topic in self.across_mode_comparisons:
            topic_only = topic[0]
            all_scores = []

            for temp in self.temperatures:
                result = self.across_mode_comparisons.get((topic_only, temp))
                if not result:
                    continue

                for mode_key, path_key, score_key in [
                    ("neutral_only", "neutral_path", "n_scores"),
                    ("multiagent", "multiagent_path", "m_scores")
                ]:
                    score_dict = result[score_key]
                    avg_score = sum(score_dict.values()) / 4
                    all_scores.append({
                        "topic": topic_only,
                        "temperature": temp,
                        "mode": mode_key,
                        "best_taxonomy_path": result[path_key],
                        "avg_score": avg_score
                    })

            if all_scores:
                best = max(all_scores, key=lambda x: x["avg_score"])
                rows.append({
                    "topic": best["topic"],
                    "temperature": best["temperature"],
                    "mode": best["mode"],
                    "best_taxonomy_path": best["best_taxonomy_path"],
                    "comparison_type": "best_overall",
                    "neutral_scores": "",
                    "multiagent_scores": "",
                    "winning_mode": ""
                })

        # save full csv
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"CSV saved to {out_path}")



    def run_all(self, topics, find_best_tax=False):
        for topic in topics:
            taxonomies = self.load_taxonomies(topic, find_best_tax)
            if taxonomies:
                self.evaluate_batch(topic, taxonomies)

if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debate_config.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    if len(sys.argv) > 1:
        topics = sys.argv[1:]
    else:
        topics = config["baseline_debate_topics"]

    # run for neutral_only
    # neutral_eval = TaxonomyEvaluator(mode="neutral_only")
    # neutral_eval.run_all(topics, find_best_tax=True)
    # neutral_eval.save_summary_csv("data/new_taxonomy/neutral_only_summary.csv")

    # run for multiagent
    # multi_eval = TaxonomyEvaluator(mode="multiagent")
    # multi_eval.run_all(topics, find_best_tax=False)
    # multi_eval.save_summary_csv("data/new_taxonomy/multiagent_summary.csv")

    # cross-mode comparisons between best taxonomies
    cross_eval = TaxonomyEvaluator()
    cross_eval.evaluate_cross_mode()
    cross_eval.save_summary_csv("data/new_taxonomy/cross_eval_summary.csv")
