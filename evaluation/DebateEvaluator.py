import os
import re
import sys
import json
import ollama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from sklearn.linear_model import LinearRegression
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(api_key=api_key)

# from openai import OpenAI
# from dotenv import load_dotenv
# load_dotenv()  
# api_key = os.getenv("OPENAI_API_KEY")

# client = OpenAI(api_key=api_key)


# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DebateEvaluator:
    def __init__(self, model, debate_group, debate_structures, num_rounds, num_iterations, metrics, evaluate_again, debate_topics):
        self.model = model
        self.num_model_calls = 3
        self.evaluate_again = evaluate_again
        self.metrics = metrics if metrics is not None else ["attitude", "final_synthesis", "repetition"]

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

            if "attitude" in self.metrics:
                self._compute_attitude_metrics(all_attitude_scores)
                # self._generate_summary_metrics(all_attitude_scores)
                topic_name = structure_topic.split("/")[-1]
                self._generate_attitude_box_plot(all_attitude_scores, topic_name)
        
        print("="*60 + "\n")


    def _compute_attitude_metrics(self, all_attitude_scores):
        metrics_df = pd.DataFrame(columns=[
            "agent",
            "num_debate_rounds",
            "num_debate_iterations",
            "mean_first_round",
            "mean_last_round",
            "mean_travel",
            "mean_iqr",
            "gradient"
        ])

        for agent, scores_per_round in all_attitude_scores.items():
            try:
                scores_per_round_np = np.array(scores_per_round)
            except ValueError:
                print("Check if `debate_iterations` is set correctly and that there are enough debate transcript files.")

            # Metrics
            mean_first_round = np.mean(scores_per_round_np[:, 0])
            mean_last_round = np.mean(scores_per_round_np[:, -1])
            mean_travel = mean_last_round - mean_first_round

            iqr_values = scipy.stats.iqr(scores_per_round_np, axis=0)
            mean_iqr = np.mean(iqr_values)

            # Linear regression to compute gradient
            X = np.arange(scores_per_round_np.shape[1]).reshape(-1, 1)
            Y = np.mean(scores_per_round_np, axis=0)
            model = LinearRegression()
            model.fit(X, Y)
            gradient = model.coef_[0]

            # Round values for cleaner CSV
            metrics_df.loc[agent] = [
                agent,
                self.num_debate_rounds,
                self.num_iterations,
                round(mean_first_round, 2),
                round(mean_last_round, 2),
                round(mean_travel, 2),
                round(mean_iqr, 2),
                round(gradient, 4)
            ]

        save_dir = self._get_relative_path(f"{'_'.join(self.debate_group)}/{self.structure_topic_path}", "data/evaluation")
        os.makedirs(save_dir, exist_ok=True)

        filename = f"metrics_{self.structure_topic_path.split('/')[-1]}_{self.num_debate_rounds}_rounds.csv"
        metrics_path = os.path.join(save_dir, filename)

        metrics_df.to_csv(metrics_path, index=False)
        print(f"Saved metrics: {metrics_path}")


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
                attitude_scores = self._evaluate_attitude_scores(transcript, topic, debate_question)
                transcript["attitude"] = attitude_scores
                result["attitude"] = attitude_scores
        elif "final_synthesis" in self.metrics:
            pass
        elif "repetition" in self.metrics:
            pass

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(transcript, f, indent=4)

        return result


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

                # if round_num > 1 and attitude_scores[agent_type][-1] is not None:
                #     prev_score = attitude_scores[agent_type][-1]
                #     score = (score + prev_score) / 2  # avg of current and previous scores

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
                    result = ollama.generate(options={"temperature":0.01}, model=self.model, prompt=prompt)
                # elif self.model == "gpt":
                #     completion = client.chat.completions.create(
                #         model="gpt-4o-mini",
                #         store=True,
                #         messages=[
                #             {"role": "user", "content": prompt}
                #         ],
                #     )
                #     result = completion.choices[0].message.content

                score = self._parse_score(result)
                if score is not None:
                    scores.append(score)
                else:
                    raise ValueError(f"Score cannot be None. This was the debate response: {response}\nPlease check the transcript file.")
            except Exception as e:
                print(f"Error with model response: {e}")

        if scores:
            return sum(scores) / len(scores)
        return None




    def _generate_attitude_judge_prompt(self, response, debate_topic):
        # attitude_scale_desc = "You should rate on a scale from 1 to 7, where: 1 = Strongly Disagree, 2 = Moderately Disagree, 3 = Slightly Disagree, " \
        #     "4 = Neutral, 5 = Slightly Agree, 6 = Moderately Agree, 7 = Strongly Agree."
        attitude_scale_desc = "You should rate on a scale from 1 (strongly disagree) to 7 (strongly agree)"

        agreement_statements = {
            "abortion": "Partial birth abortions should be banned to protect unborn children.",
            "illegal_immigration": "Immigrants take jobs from American-born workers",
            "gun_violence": "There should be stricter gun ownership laws",
            "climate_change": "The city should go ahead with go ahead with building the manufacturing plant"
        }

        # Using prompting technique from paper: https://arxiv.org/pdf/2411.15594, pg7, Section 2.1.1, Fig. 4
        attiude_judge_prompt = (
            f"Evaluate the attitude of the response from a political debate. Rate each response on the following dimension: Agreement with the statement '{agreement_statements[debate_topic]}'. {attitude_scale_desc}. Return ONLY the NUMERIC SCORE."
            f"\nDebate Response: {response}"
            f"\nAttitude Score:"
        )

        return attiude_judge_prompt


    def _parse_score(self, result):
        try:
            if self.model == "gpt":
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

        # print("X AND Y LENGTHS: ", len(x_vals), len(attitude_scores["neutral"]))
        # print(x_vals, attitude_scores["neutral"])

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
        metrics_file = os.path.join(metrics_path, f"metrics_{topic_name}_{self.num_debate_rounds}_rounds.csv")
        metrics = pd.read_csv(metrics_file)

        min_max_upper_lower_scores = {}

        for agent, rounds in attitude_scores.items():
            rounds_array = np.array(rounds)
            
            # compute min and max values per debate (across all agents for each round)
            min_max_upper_lower_scores[agent] = {
                'min': np.min(rounds_array, axis=0),  
                'max': np.max(rounds_array, axis=0),
                'upper': np.quantile(rounds_array, 0.75, axis=0),
                'lower': np.quantile(rounds_array, 0.25, axis=0)
            }

        plt.figure(figsize=(10, 5))

        # plot whiskers only
        for agent, min_max_upper_lower in min_max_upper_lower_scores.items():
            min_scores = min_max_upper_lower['min']
            max_scores = min_max_upper_lower['max']
            upper_scores = min_max_upper_lower['upper']
            lower_scores = min_max_upper_lower['lower']

            colour = self.color_mapping[agent]
            
            for i in range(len(turns)):
                # vertical line
                plt.plot([turns[i], turns[i]], [lower_scores[i], upper_scores[i]], color=colour, linewidth=2)
                
                # Top cap 
                plt.plot([turns[i] - 0.1, turns[i] + 0.1], [upper_scores[i], upper_scores[i]], color=colour, linewidth=2)
                
                # Bottom cap 
                plt.plot([turns[i] - 0.1, turns[i] + 0.1], [lower_scores[i], lower_scores[i]], color=colour, linewidth=2)

                plt.scatter(turns[i], max_scores[i], color=colour, marker="x", s=50 )  
                plt.scatter(turns[i], min_scores[i], color=colour, marker="x", s=50)  

        mean_scores = {}
        for agent in self.debate_group:
            mean_scores[agent] = np.mean(np.array(attitude_scores[agent]).T, axis=1)
        
            
        for agent in self.debate_group:
            line, = plt.plot(turns, mean_scores[agent], marker="o", linestyle="dashed", color=self.color_mapping[agent])

            # retrieve agent metrics
            agent_metrics = metrics[metrics["agent"] == agent].iloc[0]
            legend_entry = (
                f"{agent.title()}  (mTravel: {agent_metrics['mean_travel']}, "
                f"mIQR: {agent_metrics['mean_iqr']}, m: {agent_metrics['gradient']})"
            )

            # attach the formatted legend entry to the plotted line
            line.set_label(legend_entry)

        # now plt.legend() will use these labels
        plt.legend(loc="best")

        plt.xlabel("Debate Turns", fontsize=10)
        plt.ylabel(f"<- Disagree | Agree ->", fontsize=10)

        debate_question_dict = {
            "illegal_immigration": "Do immigrants take jobs from American-born workers?",
            "gun_violence": "Should there be stricter gun ownership laws?",
            "abortion": "Should partial birth abortions be banned to protect unborn children?",
            "climate_change": "Should the city go ahead with building the manufacturing plant?"
        }
        plt.title(f"Attitude Shift: {debate_question_dict[topic_name]}", fontsize=12)

        plt.legend(fontsize=8)
        plt.grid(True)

        plt.ylim(1, 7)
        plt.xlim(1, self.num_debate_rounds)
        plt.xticks(range(1, self.num_debate_rounds))

        # save plots
        plot_dir = self._get_relative_path(f"{'_'.join(self.debate_group)}/{self.structure_topic_path}", "data/evaluation")
        os.makedirs(plot_dir, exist_ok=True)

        plot_path = os.path.join(plot_dir, f"box_plot_{topic_name}_{self.num_debate_rounds}_rounds.pdf")
        plt.savefig(plot_path)
        #plt.show()
        plt.close()
        print(f"Generated box plot: {plot_path}")




    # def _generate_summary_metrics(self, all_attitude_scores):
    #     # Initialize an empty DataFrame to hold the summary metrics
    #     summary_df = pd.DataFrame(columns=["structure", "topic", "avg_convergence", "avg_travel", "avg_iqr", "avg_corr", "avg_effectiveness"])

    #     # Loop through each combination of structure and topic
    #     for structure in self.debate_structures:  # self.debate_structures is a list of debate structures (e.g., ["taxonomy_fulltree", "taxonomy_traversal"])
    #         for topic in self.debate_topics:  # self.debate_topics is a list of debate topics (e.g., ["illegal_immigration", "climate_change"])
                
    #             structure_topic = f"{structure}/{topic}"

    #             # Read the existing metrics CSV file for this structure-topic
                # metrics_path = self._get_relative_path(f"{'_'.join(self.debate_group)}/{structure_topic}", "data/evaluation")
                # metrics_file = os.path.join(metrics_path, f"metrics_{topic}_{self.num_debate_rounds}_rounds.csv")
    #             print(metrics_file)
                
    #             # Check if file exists, and load it
    #             if os.path.exists(metrics_file):
    #                 metrics_df = pd.read_csv(metrics_file)
    #             else:
    #                 print(f"Metrics file not found: {metrics_file}")
    #                 continue

    #             # Compute additional metrics based on the attitude scores in the metrics dataframe
    #             avg_convergence = 0
    #             avg_travel = 0
    #             avg_iqr = 0
    #             avg_corr = 0
    #             avg_effectiveness = 0

    #             for _, row in metrics_df.iterrows():
    #                 mean_first_round = {}
    #                 mean_last_round = {}

    #                 for agent in self.debate_group:
    #                     # Filter the row for each agent's first and last round scores
    #                     agent_data = row[row['agent'] == agent]

    #                     mean_first_round[agent] = agent_data['mean_first_round'].values[0]
    #                     mean_last_round[agent] = agent_data['mean_last_round'].values[0]

    #                 # convergence: attitude distance between closing statements of agents
    #                 closing_distance = abs(mean_last_round["republican"] - mean_last_round["neutral"])
    #                 avg_convergence += closing_distance

    #                 # opening vs closing round attiude distance (how far agents came)
    #                 opening_distance = abs(mean_first_round["republican"] - mean_first_round["neutral"])
    #                 closing_distance = abs(mean_last_round["republican"] - mean_last_round["neutral"])
    #                 delta_distance = opening_distance - closing_distance
    #                 avg_travel += delta_distance

    #                 # agent travel (total shift)
    #                 travel_neutral = mean_last_round["neutral"] - mean_first_round["neutral"]
    #                 travel_republican = mean_last_round["republican"] - mean_first_round["republican"]
    #                 travel_democrat = 
    #                 avg_travel += abs(travel_neutral) + abs(travel_republican)

    #                 # Correlation between agents (mean attitude across rounds)
    #                 neutral_scores = all_attitude_scores["neutral"]
    #                 republican_scores = all_attitude_scores["republican"]
    #                 corr = np.corrcoef(neutral_scores, republican_scores)[0, 1]
    #                 avg_corr += corr

    #                 # Effectiveness: Combined metric of travel and convergence
    #                 effectiveness = delta_distance + abs(travel_neutral) + abs(travel_republican)
    #                 avg_effectiveness += effectiveness

    #             # Average metrics for this structure-topic pair
    #             num_rows = len(metrics_df)
    #             summary_df.loc[len(summary_df)] = [
    #                 structure,
    #                 topic,
    #                 avg_convergence / num_rows,
    #                 avg_travel / num_rows,
    #                 np.mean(metrics_df["mean_iqr"]),
    #                 avg_corr / num_rows,
    #                 avg_effectiveness / num_rows,
    #             ]

    #     # Save the summary metrics to a new CSV
    #     summary_metrics_path = self._get_relative_path(f"{'_'.join(self.debate_group)}", "data/evaluation")
    #     os.makedirs(summary_metrics_path, exist_ok=True)
    #     summary_file = os.path.join(summary_metrics_path, "summary_metrics.csv")
    #     summary_df.to_csv(summary_file, index=False)
    #     print(f"Saved summary metrics to: {summary_file}")