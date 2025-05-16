import os
import json
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

METHODS = ["non_taxonomic", "taxonomic_full_tree", "taxonomic_traversal"]
METHOD_NAMES = {
    "non_taxonomic": "No Taxonomy",
    "taxonomic_full_tree": "Full Taxonomy Tree",
    "taxonomic_traversal": "Taxonomy Traversal"
}
TOPIC_NAMES = ["Immigration", "Gun Violence", "Abortion", "Economy"]  # standardised topic names
N_GRAMS = [3, 5, 10, 15, 20]
COLORS = ['#1f77b4', '#ff7f0e', '#40b37b']  # colors for each taxonomy integration type ['#1f77b4', '#ff7f0e', '#40b37b']


def clean_text(text):
    return re.sub(r'\s+', ' ', text.lower()).strip()


def compute_ngram_repetition(text, n=3):
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    counts = Counter(ngrams)
    repeated = sum(1 for count in counts.values() if count > 1)
    return repeated / len(ngrams) if len(ngrams) > 0 else 0.0


def analyse_transcript_file(filepath, n=3):
    try:
        with open(filepath) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[ERROR] Failed to process {filepath}: {e}")
        return 0.0

    text = ""
    if isinstance(data, dict):
        if 'transcript' in data:
            text = data['transcript']
        elif all(role in data for role in ["neutral", "republican", "democrat"]):
            for role in ["neutral", "republican", "democrat"]:
                for key, value in data[role].items():
                    if isinstance(value, str):
                        text += value + " "
    elif isinstance(data, list):
        text = " ".join(turn.get('content', '') for turn in data if isinstance(turn, dict))

    cleaned_text = clean_text(text)
    if not cleaned_text:
        return 0.0
    return compute_ngram_repetition(cleaned_text, n)


def get_latest_files_in_directory(directory, n=10):
    files = []
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            fp = os.path.join(directory, fname)
            try:
                mtime = os.path.getmtime(fp)
                files.append((fp, mtime))
            except OSError:
                continue
    files.sort(key=lambda x: x[1], reverse=True)
    return [fp for fp, _ in files[:n]]


def analyse_repetition_across_methods(root_dir):
    topic_results = {}
    for method in METHODS:
        method_dir = os.path.join(root_dir, method)
        if not os.path.exists(method_dir):
            print(f"[WARN] Missing directory: {method_dir}")
            continue
            
        # map original topic folders to standardised names
        topic_mapping = {}
        for topic_folder in os.listdir(method_dir):
            topic_path = os.path.join(method_dir, topic_folder)
            if os.path.isdir(topic_path):
                # assign standard topic name based on position
                idx = len(topic_mapping)
                if idx < len(TOPIC_NAMES):
                    topic_mapping[topic_folder] = TOPIC_NAMES[idx]
                else:
                    topic_mapping[topic_folder] = topic_folder  # fallback
                    
        for topic_folder, topic_name in topic_mapping.items():
            topic_path = os.path.join(method_dir, topic_folder)
            if not os.path.isdir(topic_path):
                continue
                
            if topic_name not in topic_results:
                topic_results[topic_name] = {}
            if method not in topic_results[topic_name]:
                topic_results[topic_name][method] = {n: [] for n in N_GRAMS}
                
            latest_files = get_latest_files_in_directory(topic_path, n=10)
            for filepath in latest_files:
                for n in N_GRAMS:
                    rep = analyse_transcript_file(filepath, n)
                    if rep > 0:
                        topic_results[topic_name][method][n].append(rep)
    return topic_results


def plot_repetition_results(topic_results, output_path="data/ngram_repetition_comparison.pdf"):
    # convert results to DataFrame
    data = []
    for topic, methods in topic_results.items():
        for method, ngrams in methods.items():
            for n, values in ngrams.items():
                data.extend([{
                    "Topic": topic,
                    "Method": METHOD_NAMES[method],
                    "N-gram Size": n,
                    "Repetition": val
                } for val in values])
    
    df = pd.DataFrame(data)
    if df.empty:
        print("[ERROR] No data to plot")
        return

    # ensure topics are plotted in the correct order
    df['Topic'] = pd.Categorical(df['Topic'], categories=TOPIC_NAMES, ordered=True)
    df = df.sort_values('Topic')

    # visual settings
    sns.set(style="whitegrid", font_scale=1.1)
    topics = df['Topic'].unique()
    n_topics = len(topics)
    
    # fix color mapping for methods
    method_order = ["No Taxonomy", "Full Taxonomy Tree", "Taxonomy Traversal"]
    method_colors = {
        "No Taxonomy": '#1f77b4',  # blue
        "Full Taxonomy Tree": '#ff7f0e',  # orange
        "Taxonomy Traversal": '#40b37b',  # green
    }
    
    fig, axes = plt.subplots(
        nrows=n_topics, 
        figsize=(12, 4 * n_topics),
        squeeze=False
    )
    axes = axes.flatten()

    for idx, (topic, ax) in enumerate(zip(TOPIC_NAMES, axes)):  # plot in predefined order
        if topic not in df['Topic'].unique():
            continue
            
        topic_df = df[df['Topic'] == topic]
        
        # create grouped bar plot
        sns.barplot(
            data=topic_df,
            x="N-gram Size",
            y="Repetition",
            hue="Method",
            hue_order=method_order,  # <-- enforce order for the debate structures
            ax=ax,
            palette=method_colors,
            ci='sd',
            capsize=0.1,
            errwidth=1.5
        )
        
        ax.set_title(f"Topic: {topic}", pad=20)
        ax.set_xlabel("N-gram Size", labelpad=15)
        ax.set_ylabel("Repetition Rate", labelpad=15)
        ax.legend(
            title="Generation Method",
            bbox_to_anchor=(1.05, 1),
            loc='upper left'
        )
        
        # remove spines for cleaner look
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis='y', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Visualization saved to {output_path}")


if __name__ == "__main__":
    root_dir = "data/debate_data/neutral_republican_democrat"
    results = analyse_repetition_across_methods(root_dir)
    plot_repetition_results(results)