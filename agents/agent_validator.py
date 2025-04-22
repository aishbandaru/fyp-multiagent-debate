import sys
import os
import re
import csv
import json
import ollama
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.DebateAgent import DebateAgent


MODEL = "gemini-2.0-flash-lite"
AGENT_NAMES = {"neutral": "Sam", "republican": "Alex", "democrat": "Taylor", "republican2": "Riley", "republican3": "Morgan", "democrat2": "Quinn", "democrat3": "Drew"}


# total questions: 45 (3 questions in each of 15 political issues)
INTERVIEW_QUESTIONS = quiz_questions = {
    "Immigration": [
        "Should illegal immigrants have access to government-subsidized healthcare?",
        "Should the U.S. build a wall along the southern border?",
        "Should undocumented immigrants be offered in-state tuition rates at public colleges within their residing state?"
    ],
    "Technology": [
        "Should the government implement stricter regulations on the use of cryptocurrencies?",
        "Should the government mandate that large tech companies share their algorithms with regulators?",
        "Should artists be held to the same reporting and disclosure requirements as hedge funds, mutual funds, and public companies when selling their artwork?"
    ],
    "National Security": [
        "Should the government use facial recognition technology for mass surveillance to enhance public safety?",
        "Should the President be able to authorize military force against Al-Qaeda without Congressional approval?",
        "Should the US assassinate suspected terrorists in foreign countries?"
    ],
    "Criminal Justice": [
        "Should funding for local police departments be redirected to social and community based programs?",
        "Should police departments be allowed to use military grade equipment?",
        "Should drug traffickers receive the death penalty?"
    ],
    "Electoral": [
        "Should the minimum voting age be lowered?",
        "Should the electoral college be abolished?",
        "Should a photo ID be required to vote?"
    ],
    "Environment": [
        "Should the government increase environmental regulations to prevent climate change?",
        "Should the U.S. withdraw from the Paris Climate Agreement?",
        "Do you support the use of hydraulic fracking to extract oil and natural gas resources?"
    ],
    "Healthcare": [
        "Should the government require employees of large businesses to be vaccinated from COVID?",
        "Should the federal government increase funding of health care for low income individuals (Medicaid)?",
        "Do you support the Patient Protection and Affordable Care Act (Obamacare)?"
    ],
    "Transportation": [
        "Should the government implement stricter emissions standards for diesel vehicles?",
        "Should the government impose stricter fuel efficiency standards on vehicles?",
        "Should the government require all new cars to be electric or hybrid by a certain date?"
    ],
    "Economic": [
        "Should employers be required to pay men and women the same salary for the same job?",
        "Do you believe labor unions help or hurt the economy?",
        "Should the technology of our financial system transition to a decentralized protocol, that is not owned or controlled by any corporation, similar to the internet?"
    ],
    "Science": [
        "Should producers be required to label genetically engineered foods (GMOs)?",
        "Should the government require children to be vaccinated for preventable diseases?",
        "Should the government allow the commercialization of lab-grown meat?"
    ],
    "Education": [
        "Should the federal government pay for tuition at four-year colleges and universities?",
        "Do you support increasing taxes for the rich in order to reduce interest rates for student loans?",
        "Should critical race theory be taught in K-12 education?"
    ],
    "Domestic Policy": [
        "Should there be more restrictions on the current process of purchasing a gun?",
        "Should victims of gun violence be allowed to sue firearms dealers and manufacturers?",
        "Should teachers be allowed to carry guns at school?"
    ],
    "Social": [
        "What is your stance on abortion?",
        "Should the government continue to fund Planned Parenthood?",
        "Should health insurance providers be required to offer free birth control?"
    ],
    "Housing": [
        "Should the government implement rent control policies to limit the amount landlords can charge for rent?",
        "Should the government incentivize the construction of affordable housing?",
        "Should homeless individuals, that have refused available shelter or housing, be allowed to sleep or encamp on public property?"
    ],
    "Foreign Policy": [
        "Should the government increase or decrease military spending?",
        "Should the United States provide military supplies and funding to Ukraine?",
        "Should the U.S. remain in the United Nations?"
    ]
}


def load_personas(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data.get("generated_personas", {})


def create_agents(personas, agent_names):
    agents = []
    for affiliation, configs in personas.items():
        if affiliation not in ["democrat2", "democrat3", "republican2", "republican3"]:
            agent_name = agent_names[affiliation]
            agents.append(DebateAgent(identifier=affiliation, name=agent_name, model=MODEL, party=None, leaning=None, temperature=0.3))
    return agents


def create_csv(filename, header):  # for saving interview responses
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)


def append_to_csv(filename, data):  # for updating csv (to save results incrementally)
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def interview_agents(agents, questions, csv_filename):
    csv_data = []
    for i, agent in enumerate(agents):
        print(f"Currently processing Agent number {i + 1} out of {len(agents)}")
        print(f"Agent: {agent.name}, Model: {agent.model}, Affiliation: {agent.identifier.title()}")
        
        agent.generate_prompt(use_extended_personas=True)

        for question in questions["all"]:
            question_prompt = f"Answer the question from your political perspective in 50 words or less: {question}"
            response = agent.respond(question_prompt, conversation="")

            print(question)
            print(f"{agent.name}: {response}")
            
            csv_data.append([
                agent.name,
                agent.identifier.title(),
                question,
                response
            ])
        
        if i % 10 == 0 or i == len(agents) - 1:  # save every 10 agents
            append_to_csv(csv_filename, csv_data)
            csv_data.clear()
            print(f"Processed and saved {i + 1} agents so far...")


def start_interivews():
    base_path = os.path.dirname(os.path.abspath(__file__))
    personas_file = os.path.join(base_path, "extended_personas.json")
    csv_filename = os.path.join(base_path, "agent_validation.csv")

    csv_header = ["Agent Name", "Affiliation", "Question", "Response"]
    
    personas = load_personas(personas_file)
    agents = create_agents(personas, AGENT_NAMES)
    create_csv(csv_filename, csv_header)
    interview_agents(agents, INTERVIEW_QUESTIONS, csv_filename)
    
    print(f"CSV file '{csv_filename}' has been created successfully.")



def evaluate_response(question, response, affiliation, gender, age):

    if affiliation == "neutral":
        political_affiliation = "Neutral American"
    elif affiliation == "republican":
        political_affiliation = "Republican American"
    elif affiliation == "democrat":
            political_affiliation = "Democrat American"

    prompt = f"""
    Does the response reflect the typical views of the political affiliation on the question '{question}'? Answer "Yes" or "No".
    If your answer is "No", explain your reasoning in less than 50 words.

    Response: {response}
    Political Affiliation: {political_affiliation}
    """

    try:
        llm_response = ollama.chat(model="mistral:7b", messages=[{"role": "user", "content": prompt}])
        llm_text = llm_response["message"]["content"]

        # print(prompt)
        # print("\n\n",llm_text)

        # Extract answer: look for "Yes" or "No" at beginning or after a label
        answer_match = re.search(r'\b(?:Answer\s*:)?\s*(Yes|No)\b', llm_text, re.IGNORECASE)
        evaluation = answer_match.group(1).capitalize() if answer_match else "Error: Unable to parse"

        # Extract explanation if "No"
        if evaluation.lower() == "no":
            explanation_match = re.search(r'(?:Explanation\s*:)?\s*(.+)', llm_text, re.IGNORECASE | re.DOTALL)
            explanation = explanation_match.group(1).strip() if explanation_match else "Error: No explanation found"
        else:
            explanation = ""

        # print(evaluation, explanation)
        return evaluation, explanation
    except Exception as e:
        return "Error", str(e)


def evaluate_interviews():
    base_path = os.path.dirname(os.path.abspath(__file__))
    input_csv_filename = os.path.join(base_path, "agent_validation.csv")
    output_csv_filename = os.path.join(base_path, "agent_validation_evaluated.csv")

    batch_size = 100

    with open(input_csv_filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        fieldnames = reader.fieldnames + ["Evaluation", "Explanation"]

        for i, row in enumerate(rows):
            if "Evaluation" in row and row["Evaluation"]:  # skip already processed rows
                continue

            evaluation, explanation = evaluate_response(row["Question"], row["Response"], row["Affiliation"])
            row["Evaluation"] = evaluation
            row["Explanation"] = explanation

            # save every `batch_size` rows to prevent data loss
            if (i + 1) % batch_size == 0 or i == len(rows) - 1:
                with open(output_csv_filename, mode="w", newline="", encoding="utf-8") as output_csv:
                    writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                print(f"Saved progress: {i + 1}/{len(rows)} entries processed.")



def load_and_prepare_data(simple_csv, extended_csv):
    # Load both datasets
    df_simple = pd.read_csv(simple_csv)
    df_extended = pd.read_csv(extended_csv)
    
    # Add persona type identifier
    df_simple['Persona Type'] = 'Simple'
    df_extended['Persona Type'] = 'Extended'
    
    # Combine datasets
    df_combined = pd.concat([df_simple, df_extended])
    
    # Convert evaluations to numerical scores
    df_combined['Score'] = df_combined['Evaluation'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )
    
    # Map questions to their categories
    question_to_category = {}
    for category, questions in INTERVIEW_QUESTIONS.items():
        for q in questions:
            question_to_category[q] = category
    
    df_combined['Category'] = df_combined['Question'].map(question_to_category)
    
    # Calculate mean alignment by category and persona type
    results = df_combined.groupby(['Persona Type', 'Category'])['Score'].mean().reset_index()
    
    return results

def create_radar_chart(data, output_file="radar_comparison.png"):
    # Prepare data for radar chart
    categories = list(INTERVIEW_QUESTIONS.keys())
    N = len(categories)
    
    # Create angles for radar chart
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Set up the figure
    plt.figure(figsize=(12, 12))
    ax = plt.subplot(111, polar=True)
    
    # Set the first axis to be at the top
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw one axe per variable and add labels
    plt.xticks(angles[:-1], categories, size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["25%", "50%", "75%", "100%"], color="grey", size=10)
    plt.ylim(0, 1)
    
    # Plot data for each persona type
    colors = ['#1f77b4', '#ff7f0e']  # Blue for simple, orange for extended
    for i, persona_type in enumerate(['Simple', 'Extended']):
        subset = data[data['Persona Type'] == persona_type]
        
        # Ensure categories are in correct order
        subset = subset.set_index('Category').reindex(categories).reset_index()
        values = subset['Score'].values.tolist()
        values += values[:1]  # Close the circle
        
        ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', 
                label=f'{persona_type} Personas')
        ax.fill(angles, values, color=colors[i], alpha=0.25)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Add title
    plt.title('Political Alignment Comparison:\nSimple vs Extended Personas', 
              size=15, y=1.15)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Radar chart saved as {output_file}")


def visualise_evaluation():
    # Paths to your CSV files
    simple_csv = "path/to/simple_personas_results.csv"
    
    extended_csv = "path/to/extended_personas_results.csv"
    
    # Verify files exist
    if not os.path.exists(simple_csv):
        raise FileNotFoundError(f"Simple personas CSV not found at {simple_csv}")
    if not os.path.exists(extended_csv):
        raise FileNotFoundError(f"Extended personas CSV not found at {extended_csv}")
    
    # Process data and create visualization
    prepared_data = load_and_prepare_data(simple_csv, extended_csv)
    create_radar_chart(prepared_data)


if __name__ == "__main__":
    start_interivews()
    # evaluate_interviews()
    # visualise_evaluation()
