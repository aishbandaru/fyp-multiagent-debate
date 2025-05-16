import os
import re
import sys
import json
import scipy
import ollama
import numpy as np
import pandas as pd
import multiprocessing
from openai import OpenAI
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import google.generativeai as genai
from multiprocessing import Manager, Pool
from sklearn.linear_model import LinearRegression
from google.generativeai.types import GenerationConfig

from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from nltk.corpus import stopwords
from scipy import stats
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer


genai.configure(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DebateEvaluator:
    def __init__(self, model, debate_group, debate_structures, num_rounds, num_iterations, metrics, evaluate_again, debate_topics):
        self.model = model
        self.num_model_calls = 2
        self.evaluate_again = evaluate_again
        self.metrics = metrics if metrics is not None else ["attitude", "final_synthesis", "wordcloud"]

        self.debate_group = debate_group.split("_")
        self.debate_structures = debate_structures
        self.debate_topics = debate_topics

        self.num_debate_rounds = num_rounds   # TODO: DON'T ASSUME - assuming all transcripts in this dir have the same number of rounds 
        self.num_iterations = num_iterations

        self.transcript_filename = None  # used for saving plots
        self.structure_topic_path = None # used for saving plots
        self.color_mapping = {
            'neutral': 'tan',
            'republican': 'red',
            'democrat': 'blue',
            'republican2': 'firebrick',
            'democrat2': 'navy',
            'republican3': 'lightcoral',
            'democrat3': 'skyblue',
        }

        import nltk
        nltk.download('stopwords')
        self.stop_words = set(stopwords.words('english'))


    def _load_transcript(self, filename):
        self.transcript_filename = filename
        with open(filename, "r") as file:
            return json.load(file)


    def evaluate_debates(self, base_transcripts_path, debate_transcripts):

        for structure_topic, transcript_list in debate_transcripts.items():
            print("\n" + "="*60)
            print(f"Evaluating: {structure_topic}\n")
            all_attitude_scores = {}
            self.structure_topic_path = structure_topic

            if "attitude" in self.metrics:
                for agent in self.debate_group:
                    all_attitude_scores[agent] = [[] for _ in range(self.num_iterations)]

                # loop through each transcript to get attitude scores
                for debate, transcript in enumerate(transcript_list):
                    transcript_path = os.path.join(base_transcripts_path, structure_topic, transcript)
                    result = self.evaluate_transcript(transcript_path)  # evaluate transcript

                    print("Evaluation scores: ", result)

                    if "attitude" in self.metrics:
                        scores = result.get("attitude", None)
                        for agent in self.debate_group:
                            all_attitude_scores[agent][debate] = scores[agent]

                self._compute_attitude_metrics(all_attitude_scores)
                # self._generate_summary_metrics(all_attitude_scores)
                topic_name = structure_topic.split("/")[-1]
                self._generate_attitude_box_plot(all_attitude_scores, topic_name)

        print("="*60 + "\n")


    def perform_statistical_testing(self, avg_similarities):
        # perform ANOVA to check if means of cosine similarities are different across structures
        anova_result = stats.f_oneway(*avg_similarities.values())
        
        # perform Levene's test to check if variances are equal across structures
        levene_result = stats.levene(*avg_similarities.values())
        
        print(f"ANOVA result: F-statistic = {anova_result.statistic}, p-value = {anova_result.pvalue}")
        print(f"Levene's test result: Statistic = {levene_result.statistic}, p-value = {levene_result.pvalue}")
        
        return anova_result, levene_result


    def _compute_attitude_metrics(self, all_attitude_scores):
        max_rounds = max(len(scores_per_round[0]) for scores_per_round in all_attitude_scores.values())
        max_iters = max(len(scores_per_round) for scores_per_round in all_attitude_scores.values())

        base_columns = [
            "agent",
            "num_debate_rounds",
            "num_debate_iterations",
            "mean_first_round",
            "mean_last_round",
            "mean_travel",
            "mean_iqr",
            "gradient"
        ]

        # create columns for average scores per round (wide summary csv with mean metrics)
        round_avg_columns = [f"avg_round_{i+1}" for i in range(max_rounds)]

        metrics_df = pd.DataFrame(columns=base_columns + round_avg_columns)

        # DataFrame for long format csv with detailed scores: one row per agent-round-iteration score
        long_format_rows = []

        for agent, scores_per_round in all_attitude_scores.items():
            try:
                scores_per_round_np = np.array(scores_per_round)
            except ValueError:
                print(f"Check `debate_iterations` or data for agent {agent}. Skipping.")
                continue

            # summary metrics calculation
            mean_first_round = np.mean(scores_per_round_np[:, 0])
            mean_last_round = np.mean(scores_per_round_np[:, -1])
            mean_travel = mean_last_round - mean_first_round

            iqr_per_round = scipy.stats.iqr(scores_per_round_np, axis=1)
            mean_iqr = np.mean(iqr_per_round)

            # linear regression for gradient
            X = np.arange(scores_per_round_np.shape[1]).reshape(-1, 1)
            Y = np.mean(scores_per_round_np, axis=0)
            model = LinearRegression()
            model.fit(X, Y)
            gradient = model.coef_[0]

            # average scores per round
            avg_scores_per_round = np.mean(scores_per_round_np, axis=0)
            avg_scores_rounded = [round(x, 2) for x in avg_scores_per_round]
            if len(avg_scores_rounded) < max_rounds:
                avg_scores_rounded += [float('nan')] * (max_rounds - len(avg_scores_rounded))

            # build summary row
            metrics_df.loc[agent] = [
                agent,
                self.num_debate_rounds,
                self.num_iterations,
                round(mean_first_round, 2),
                round(mean_last_round, 2),
                round(mean_travel, 2),
                round(mean_iqr, 2),
                round(gradient, 4),
                *avg_scores_rounded
            ]

            # build long-format data: iterate over rounds and iterations
            # scores_per_round_np shape: (num_iterations, num_rounds)
            for iteration_idx in range(scores_per_round_np.shape[0]):
                for round_idx in range(scores_per_round_np.shape[1]):
                    score = scores_per_round_np[iteration_idx, round_idx]
                    long_format_rows.append({
                        "agent": agent,
                        "round": round_idx + 1,
                        "iteration": iteration_idx + 1,
                        "score": score
                    })

        long_format_df = pd.DataFrame(long_format_rows)

        # save both files
        save_dir = self._get_relative_path(f"{'_'.join(self.debate_group)}/{self.structure_topic_path}", "data/evaluation")
        os.makedirs(save_dir, exist_ok=True)

        summary_filename = f"metrics_mean_{self.structure_topic_path.split('/')[-1]}_{self.num_debate_rounds}_rounds.csv"
        summary_path = os.path.join(save_dir, summary_filename)
        metrics_df.to_csv(summary_path, index=False)
        print(f"Saved summary metrics: {summary_path}")

        long_filename = f"metrics_long_{self.structure_topic_path.split('/')[-1]}_{self.num_debate_rounds}_rounds.csv"
        long_path = os.path.join(save_dir, long_filename)
        long_format_df.to_csv(long_path, index=False)
        print(f"Saved long-format detailed scores: {long_path}")


    def evaluate_transcript(self, filename):
        transcript = self._load_transcript(filename)
        result = {}
        topic = transcript["topic"]
        debate_question = transcript["debate_question"]
    
        num_agents = len(self.debate_group)
        if num_agents not in [2, 3, 4]:
            raise ValueError("The evaluation data in JSON file must contain exactly 2, 3 or 4 agents.")

        if "attitude" in self.metrics:
            # check if scores already computed
            attitude_scores =  transcript.get("attitude", None)
            if attitude_scores is not None and not self.evaluate_again:
                result["attitude"] = attitude_scores
            else:
                attitude_scores = self._evaluate_attitude_scores_parallel(transcript, topic, debate_question)
                transcript["attitude"] = attitude_scores
                result["attitude"] = attitude_scores
        elif "final_synthesis" in self.metrics:
            pass
        elif "repetition" in self.metrics:
            pass

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=4)

        return result
    

    def _evaluate_attitude_scores_parallel(self, transcript, topic_name, debate_question):
        # prep all arguments for parallel processing
        args = [(transcript, topic_name, round_num) for round_num in range(self.num_debate_rounds)]
        
        # process rounds in parallel
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            round_results = pool.starmap(self._evaluate_single_round, args)
        
        # reconstruct attitude_scores dict from the results
        attitude_scores = {agent: [] for agent in self.debate_group}
        for round_result in round_results:
            for agent, score in round_result.items():
                attitude_scores[agent].append(score)
        
        self._generate_attitude_plot(attitude_scores, debate_question)
        print(f"\nCompleted attitude evaluation of debate topic {topic_name} with question: {debate_question}.\n")
        return attitude_scores

    def _evaluate_single_round(self, transcript, debate_topic, round_num):
        round_scores = {}
        round_num_label = f"round_{round_num}"
        
        for agent_type in self.debate_group:
            response = transcript.get(agent_type, {}).get(round_num_label)
            if response:
                score = self._get_llm_attitude_score(response, debate_topic)
                round_scores[agent_type] = score if score is not None else 4
            else:
                round_scores[agent_type] = None
        
        return round_scores


    def _evaluate_attitude_scores(self, transcript, topic_name, debate_question):
        attitude_scores = {agent: [] for agent in self.debate_group}

        for round_num in range(0, self.num_debate_rounds):
            self._evaluate_round(transcript, attitude_scores, topic_name, round_num)
        
        self._generate_attitude_plot(attitude_scores, debate_question)
        print(f"\nCompleted attitude evaluation of debate topic {topic_name} with question: {debate_question}.\n")
        return attitude_scores


    def _evaluate_round(self, transcript, attitude_scores, debate_topic, round_num):
        for agent_type in self.debate_group:
            round_num_label = f"round_{round_num}"
            response = transcript.get(agent_type, {}).get(round_num_label)

            if response:
                score = self._get_llm_attitude_score(response, debate_topic)

                attitude_scores[agent_type].append(score if score is not None else 4)
            else:
                attitude_scores[agent_type].append(None)


    def _get_llm_attitude_score(self, response, debate_topic):
        # get attitude score using LLM for multiple attempts
        scores = []
        for _ in range(self.num_model_calls):
            prompt = self._generate_attitude_judge_prompt(response, debate_topic)
            try:
                if (self.model != "gpt") and ("gemini" not in self.model):
                    result = ollama.generate(options={"temperature":0}, model=self.model, prompt=prompt)
                elif self.model == "gpt":
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        # store=True,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0
                    )
                    result = completion.choices[0].message.content
                elif self.model == "gemini":
                    model = genai.GenerativeModel("gemini-2.0-flash")
                    generation_config = GenerationConfig(
                        temperature=0,
                        max_output_tokens=1024,
                    )
                    response = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    result = response.text

                score = self._parse_score(result)
                if score is not None:
                    scores.append(score)
                else:
                    raise ValueError(f"Score cannot be None. This was the debate response: {result}\nPlease check the transcript file.")
            except Exception as e:
                print(f"Error with model response: {e}")

        if scores:
            return sum(scores) / len(scores)
        return None


    def _generate_attitude_judge_prompt(self, response, debate_topic):
        attitude_scale_desc = "You should rate on a scale from 1 to 7, where: 1 = Strongly Disagree, 2 = Moderately Disagree, 3 = Slightly Disagree, " \
            "4 = Neutral, 5 = Slightly Agree, 6 = Moderately Agree, 7 = Strongly Agree."

        agreement_statements = {
            "illegal immigration": "Immigrants take jobs from American-born workers.",
            "gun violence": "There should be stricter gun ownership laws.",
            "abortion": "Partial-birth abortions should be banned to protect unborn children.",
            "economy": "Congress should raise taxes on the wealthy to reduce the federal budget deficit.",
            "religion": "Businesses should be allowed to deny service based on religious beliefs."
        }

        # using prompting technique from paper, A Survey on LLM-as-a-Judge: https://arxiv.org/pdf/2411.15594, pg7, Section 2.1.1, Fig. 4
        attiude_judge_prompt = (
            f"Evaluate the attitude of the response from a political debate. Rate each response on one dimension: Agreement with the statement '{agreement_statements[debate_topic]}'. {attitude_scale_desc}. Return ONLY the NUMERIC SCORE."
            f"\nDebate Response: {response}"
            f"\nAttitude Score:"
        )

        return attiude_judge_prompt


    def _parse_score(self, result):
        try:
            if self.model in ["gpt", "gemini"]:
                digit = re.findall(r'\d', result.strip())
            else:
                digit = re.findall(r'\d', result["response"].strip())
            score = int(digit[0])
            return max(1, min(7, score))
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Unable to parse model response on the attitude score. Response:\n{result}")
            return None


    def _generate_attitude_plot(self, attitude_scores, debate_question):

        plt.figure(figsize=(10, 5))
        x_vals = list(range(1, self.num_debate_rounds + 1))

        for agent in self.debate_group:
            color = self.color_mapping.get(agent.split('_')[0].lower(), 'gray')
            label = f"{agent.title()} Attitude"
            plt.plot(x_vals, attitude_scores[agent], marker="o", label=label, color=color)

        plt.xlabel("Debate Round")
        plt.ylabel("Attitude Score")
        plt.title(f"Attitude Shift Over Debate: {debate_question}")
        plt.legend()
        plt.grid(True)
        plt.ylim(1, 7)  # attitude score from 1 to 7
        plt.xlim(1, self.num_debate_rounds)  # restrict x-axis to num_debate rounds to analyse
        plt.xticks(range(1, self.num_debate_rounds))

        plot_dir = self._get_relative_path(f"{'_'.join(self.debate_group)}/{self.structure_topic_path}", "data/evaluation")
        os.makedirs(plot_dir, exist_ok=True)

        datetime_match = re.search(r'transcript_(\d{8}_\d{6})\.json', self.transcript_filename)
        if datetime_match:
            timestamp = datetime_match.group(1)

        plot_path = os.path.join(plot_dir, f"{self.structure_topic_path.split('/')[-1]}_{self.num_debate_rounds}_rounds_{timestamp}.pdf")
        plt.savefig(plot_path)
        plt.close()


    def _get_relative_path(self, filename, folder="data"):
        # to enable running evaluation from root folder or evaluation subfolder
        if os.path.basename(os.getcwd()) == folder:
            return filename
        return os.path.join(folder, filename)



    def _generate_attitude_box_plot(self, attitude_scores, topic_name):  
        turns = np.array(range(1, self.num_debate_rounds + 1), dtype=np.float32)
        
        # read metrics CSV
        metrics_path = self._get_relative_path(f"{'_'.join(self.debate_group)}/{self.structure_topic_path}", "data/evaluation")
        metrics_file = os.path.join(metrics_path, f"metrics_mean_{topic_name}_{self.num_debate_rounds}_rounds.csv")
        metrics = pd.read_csv(metrics_file)

        # create figure with adjusted height
        plt.figure(figsize=(10.5, 7.0))  # height for 2-line legend
        
        # plot all elements
        for agent in self.debate_group:
            scores_array = np.array(attitude_scores[agent])
            q1 = np.percentile(scores_array, 25, axis=0)
            q3 = np.percentile(scores_array, 75, axis=0)
            min_vals = np.min(scores_array, axis=0)
            max_vals = np.max(scores_array, axis=0)
            color = self.color_mapping[agent]
            
            # for whiskers
            # for i, turn in enumerate(turns):
            #     plt.plot([turn, turn], [q1[i], q3[i]], 
            #             color=color, linewidth=2, alpha=0.7)
            #     plt.plot([turn-0.1, turn+0.1], [q1[i], q1[i]], 
            #             color=color, linewidth=2, alpha=0.7)
            #     plt.plot([turn-0.1, turn+0.1], [q3[i], q3[i]], 
            #             color=color, linewidth=2, alpha=0.7)
            #     plt.scatter(turn, min_vals[i], color=color, marker='x', s=50, zorder=3)
            #     plt.scatter(turn, max_vals[i], color=color, marker='x', s=50, zorder=3)

        # plot mean lines
        mean_lines = []
        for agent in self.debate_group:
            mean_scores = np.mean(np.array(attitude_scores[agent]), axis=0)
            line = plt.plot(turns, mean_scores, linestyle="-",
                        linewidth=2.5, color=self.color_mapping[agent], zorder=2)
            mean_lines.append(line[0])

        # create compact legend labels
        legend_labels = []
        for agent in self.debate_group:
            agent_metrics = metrics[metrics["agent"] == agent].iloc[0]
            legend_labels.append(
                f"{agent.title()} (travel: {agent_metrics['mean_travel']:.2f} " + 
                f"IQR: {agent_metrics['mean_iqr']:.2f} m: {agent_metrics['gradient']:.2f})"
            )

        # calculate optimal columns for 2-line legend
        n_agents = len(self.debate_group)
        ncol = n_agents // 2 + n_agents % 2  # split into 2 roughly equal lines

        # put legend below current axis
        leg = plt.legend(mean_lines, legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=ncol,
            fontsize=11.5,
            framealpha=0.9,
            handlelength=1.5,
            columnspacing=1.0,
            handletextpad=0.5,
            labelspacing=0.5,
            borderpad=0.5)

        # adjust plot margins
        plt.subplots_adjust(bottom=0.25 + 0.05*n_agents)  # dynamic bottom margin

        # formatting
        plt.xlabel("Debate Round", fontsize=11.5)
        plt.ylabel("Attitude Score (1-7)", fontsize=11.5)
        
        debate_question_dict = {
            "illegal_immigration": "Do immigrants take jobs from American-born workers?",
            "gun_violence": "Should there be stricter gun ownership laws?",
            "abortion": "Should partial birth abortions be banned to protect unborn children?",
            "economy": "Should Congress raise taxes on the wealthy to reduce the federal budget deficit?"
        }
        debate_structures = {
            "taxonomic_full_tree": "Full Taxonomy Tree",
            "taxonomic_traversal": "Taxonomy Traversal",
            "non_taxonomic": "No Taxonomy"
        }
        plt.title(f"{debate_structures[self.structure_topic_path.split('/')[0]]}: {debate_question_dict.get(topic_name, topic_name)}", 
                fontsize=12, pad=15)
        
        plt.grid(True, alpha=0.3)
        plt.ylim(1, 7)
        plt.xlim(0.5, self.num_debate_rounds+0.5)
        plt.xticks(range(1, self.num_debate_rounds+1))

        # save plot
        plot_dir = self._get_relative_path(f"{'_'.join(self.debate_group)}/{self.structure_topic_path}", "data/evaluation")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"box_plot_{topic_name}_{self.num_debate_rounds}_rounds.pdf")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated box plot: {plot_path}")

