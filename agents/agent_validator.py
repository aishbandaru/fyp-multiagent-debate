import sys
import os
import re
import csv
import json
import ollama
import numpy as np
import pandas as pd
from openai import OpenAI
import matplotlib.pyplot as plt

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.DebateAgent import DebateAgent


BASE_PATH = os.path.dirname(os.path.abspath(__file__))
AGENT_MODEL = "gemini-2.0-flash"
AGENT_NAMES = {"neutral": "Sam", "republican": "Alex", "democrat": "Taylor", "republican2": "Riley", "republican3": "Morgan", "democrat2": "Quinn", "democrat3": "Drew"}
EVALUATION_MODEL = "gpt-4.1-2025-04-14"  # options: "mistral:7b", "gpt-4o-mini", "gpt-4.1-2025-04-14"


INTERVIEW_QUESTIONS = {
    "Immigration": [
        "Should illegal immigrants have access to government-subsidized healthcare?",
        "Should the U.S. build a wall along the southern border?",
        "Should undocumented immigrants be offered in-state tuition rates at public colleges within their residing state?",
        "Should sanctuary cities receive federal funding?",
        "Should local law enforcement be allowed to detain illegal immigrants for minor crimes and transfer them to federal immigration authorities?",
        "Should children of illegal immigrants be granted legal citizenship?",
        "Should the U.S. increase restrictions on its current border security policy?",
        "Should immigrants be deported if they commit a serious crime?",
        "Should immigrants from high risk countries be banned from entering the country until the government improves its ability to screen out potential terrorists?",
        "Should immigrants be required to learn English?"
    ],
    "Technology": [
        "Should the government implement stricter regulations on the use of cryptocurrencies?",
        "Should the government mandate that large tech companies share their algorithms with regulators?",
        "Should artists be held to the same reporting and disclosure requirements as hedge funds, mutual funds, and public companies when selling their artwork?",
        "Should the government impose stricter regulations on the collection and use of personal data by companies?",
        "Should the government regulate artificial intelligence (AI) to ensure ethical use?",
        "Should citizens be allowed to secure their money in self-hosted digital wallets that the government can monitor but not control?"
    ],
    "National Security": [
        "Should the government use facial recognition technology for mass surveillance to enhance public safety?",
        "Should the President be able to authorize military force against Al-Qaeda without Congressional approval?",
        "Should the US assassinate suspected terrorists in foreign countries?",
        "Should the President mobilize the U.S. military against Mexican Drug Cartels?",
        "Should the government implement a national identification system to enhance security and prevent fraud?",
        "Should the government require tech companies to provide backdoor access to encrypted communications for national security purposes?",
        "Should the government invest in artificial intelligence (AI) for defense applications?",
        "Should the government ban its citizens from using cross-border payment methods (like crypto) to send money to relatives in OFAC sanctioned countries (Palestine, Iran, Cuba, Venezuela, Russia, and North Korea)?"
    ],
    "Criminal Justice": [
        "Should funding for local police departments be redirected to social and community based programs?",
        "Should police departments be allowed to use military grade equipment?",
        "Should drug traffickers receive the death penalty?",
        "Should prisons ban the use of solitary confinement for juveniles?",
        "Do you support qualified immunity for police officers?",
        "Do you support mandatory minimum prison sentences for people charged with drug possession?",
        "Should convicted criminals have the right to vote?",
        "Should non-violent prisoners be released from jail in order to reduce overcrowding?",
        "Do you support limiting police unions collective bargaining power for cases involving misconduct?",
        "Should the government implement restorative justice programs as an alternative to incarceration?",
        "Should the penalty for traffic violations depend on the driver's income?",
        "Should police officers be required to wear body cameras?",
        "Should the government hire private companies to run prisons?",
        "Should AI be used to make decisions in criminal justice systems?"
    ],
    "Electoral": [
        "Should the minimum voting age be lowered?",
        "Should the electoral college be abolished?",
        "Should a photo ID be required to vote?",
        "Should a politician, who has been formerly convicted of a crime, be allowed to run for office?",
        "Should there be a limit to the amount of money a candidate can receive from a donor?",
        "Should political candidates be required to release their recent tax returns to the public?",
        "Should every voter automatically receive a mail in ballot?",
        "Should corporations, unions, and non-profit organizations be allowed to donate to political parties?",
        "Should there be a 5-year ban on White House and Congressional officials from becoming lobbyists after they leave the government?",
        "Should foreign lobbyists be allowed to raise money for American elections?",
        "Should politicians over 75 years of age have required to pass a mental competency test?"
    ],
    "Environment": [
        "Should the government increase environmental regulations to prevent climate change?",
        "Should the U.S. withdraw from the Paris Climate Agreement?",
        "Do you support the use of hydraulic fracking to extract oil and natural gas resources?",
        "Should the government give tax credits and subsidies to the wind power industry?",
        "Should drilling be allowed in the Alaska Wildlife Refuge?",
        "Should the U.S. expand offshore oil drilling?",
        "Should the government stop construction of the Dakota Access pipeline?",
        "Should disposable products (such as plastic cups, plates, and cutlery) that contain less than 50% of biodegradable material be banned?",
        "Should the government fund research into geoengineering as a way to combat climate change?",
        "Should the government build a network of electric vehicle charging stations?",
        "Should researchers be allowed to use animals in testing the safety of drugs, vaccines, medical devices, and cosmetics?",
        "Should the government invest in programs to reduce food waste?",
        "Should the government provide subsidies for companies developing carbon capture technologies?",
        "Should the federal government support EV adoption through incentives and infrastructure funding?",
        "Should the government provide subsidies to taxpayers who purchase an electric vehicle?",
        "Should cities be allowed to offer private companies economic incentives to relocate?",
        "Should the U.S. lift the moratorium on new liquefied natural gas (LNG) export licenses?"
    ],
    "Healthcare": [
        "Should the government require employees of large businesses to be vaccinated from COVID?",
        "Should the federal government increase funding of health care for low income individuals (Medicaid)?",
        "Do you support the Patient Protection and Affordable Care Act (Obamacare)?",
        "Should the government fund the World Health Organization?",
        "Should the government forgive all medical debt for Americans?",
        "Should people be required to work in order to receive Medicaid?",
        "Do you support a single-payer healthcare system?",
        "Should cities open drug 'safe havens' where people who are addicted to illegal drugs can use them under the supervision of medical professionals?",
        "Should the government regulate the prices of life-saving drugs?",
        "Should there be more or less privatization of veterans' healthcare?",
        "Should the federal government be allowed to negotiate drug prices for Medicare?",
        "Should medical boards penalize doctors who give health advice that contradicts contemporary scientific consensus?",
        "Should the government increase funding for mental health research and treatment?",
        "Should the government eliminate price caps on generic drugs?",
        "Should the government ban the promotion of products that contribute to unhealthy lifestyles for young people, such as vaping and junk food?"
    ],
    "Transportation": [
        "Should the government implement stricter emissions standards for diesel vehicles?",
        "Should the government impose stricter fuel efficiency standards on vehicles?",
        "Should the government require all new cars to be electric or hybrid by a certain date?",
        "Should the government invest in the development of smart transportation infrastructure?",
        "Should the government prioritize the maintenance and repair of existing roads and bridges over building new infrastructure?",
        "Should the government provide subsidies for the development of high-speed rail networks?",
        "Should the government require public transportation systems to be fully accessible to people with disabilities?",
        "Should the government increase spending on public transportation?",
        "Should the government regulate the development and deployment of autonomous vehicles?",
        "Should the government provide incentives for carpooling and the use of shared transportation services?",
        "Should cities implement congestion pricing to reduce traffic in busy urban areas?",
        "Should the government increase penalties for distracted driving?"
    ],
    "Economic": [
        "Should employers be required to pay men and women the same salary for the same job?",
        "Do you believe labor unions help or hurt the economy?",
        "Should the technology of our financial system transition to a decentralized protocol, that is not owned or controlled by any corporation, similar to the internet?",
        "Should the U.S. raise taxes on the rich?",
        "Should the government raise the federal minimum wage?",
        "Should the U.S. raise or lower the tax rate for corporations?",
        "Do you support a universal basic income program?",
        "Should welfare recipients be tested for drugs?",
        "Should the government make cuts to public spending in order to reduce the national debt?",
        "Should there be fewer or more restrictions on current welfare benefits?",
        "Should the government increase the tax rate on profits earned from the sale of stocks, bonds, and real estate?",
        "Should the current estate tax rate be decreased?",
        "Should the government enforce a cap on CEO pay relative to the pay of their employees?",
        "Should the government use economic stimulus to aid the country during times of recession?",
        "Should the government require businesses to pay salaried employees, making up to $46k/year, time-and-a-half for overtime hours?",
        "Should the government tax unrealized gains?",
        "Should the United States transition to a four-day workweek?",
        "Should the government break up Amazon, Facebook and Google?",
        "Should U.S. citizens be allowed to save or invest their money in offshore bank accounts?"
    ],
    "Science": [
        "Should producers be required to label genetically engineered foods (GMOs)?",
        "Should the government require children to be vaccinated for preventable diseases?",
        "Should the government allow the commercialization of lab-grown meat?",
        "Should the government fund research into genetic engineering for disease prevention and treatment?",
        "Should the government regulate the use of CRISPR technology for human genetic modifications?",
        "Should the government fund space travel?"
    ],
    "Education": [
        "Should the federal government pay for tuition at four-year colleges and universities?",
        "Do you support increasing taxes for the rich in order to reduce interest rates for student loans?",
        "Should critical race theory be taught in K-12 education?",
        "Should the federal government fund Universal preschool?",
        "Do you support Common Core national standards?",
        "Do you support charter schools?",
        "Should the government decriminalize school truancy?",
        "Should the government offer students a voucher that they can use to attend private schools?",
        "Should colleges be held financially accountable if graduates, with degrees leading to lower income jobs, default on their student loans?"
    ],
    "Domestic Policy": [
        "Should there be more restrictions on the current process of purchasing a gun?",
        "Should victims of gun violence be allowed to sue firearms dealers and manufacturers?",
        "Should teachers be allowed to carry guns at school?",
        "Should the Supreme Court be reformed to include more seats and term limits on judges?",
        "Do you support affirmative action programs?",
        "Should people on the 'no-fly list' be banned from purchasing guns and ammunition?",
        "Are you in favor of decriminalizing drug use?",
        "Should it be illegal to burn the American flag?",
        "Should Supreme Court justices be prohibited from making financial transactions with people who have a vested interest in court outcomes?",
        "Do you support the Patriot Act?",
        "Should the government be allowed to seize private property, with reasonable compensation, for public or civic use?",
        "Should the government regulate social media sites, as a means to prevent fake news and misinformation?",
        "Should the federal government institute a mandatory buyback of assault weapons?",
        "Should the redrawing of Congressional districts be controlled by an independent, non-partisan commission?",
        "Should those charged in the January 6 Capitol attack be granted Presidential pardons?",
        "Should members of Congress be allowed to trade stocks while serving in office?",
        "Should the U.S. government grant immunity to Edward Snowden?",
        "Should social media companies ban political advertising?",
        "Should the United States acquire Greenland?"
    ],
    "Social": [
        "What is your stance on abortion?",
        "Should the government continue to fund Planned Parenthood?",
        "Should health insurance providers be required to offer free birth control?",
        "Do you support the legalization of same sex marriage?",
        "Should 'gender identity' be added to anti-discrimination laws?",
        "Should transgender athletes be allowed to compete against athletes that differ from their assigned sex at birth?",
        "Should a business be able to deny service to a customer if the request conflicts with the owner's religious beliefs?",
        "Should the federal government require racial sensitivity training for employees?",
        "Should gay couples have the same adoption rights as straight couples?",
        "Should universities provide 'trigger warnings' and 'safe spaces' for students?",
        "Should people under the age of 18 years old be able to receive gender-transition treatments?",
        "Do you support the death penalty?",
        "Should businesses be required to have women on their board of directors?",
        "Should states be allowed to display the Confederate flag on government property?",
        "Should the government support a separation of church and state by removing references to God on money, federal buildings, and national monuments?",
        "Should DEI programs be eliminated from federal government agencies?",
        "Should the government officially recognize only two biological sexes?"
    ],
    "Housing": [
        "Should the government implement rent control policies to limit the amount landlords can charge for rent?",
        "Should the government incentivize the construction of affordable housing?",
        "Should homeless individuals, that have refused available shelter or housing, be allowed to sleep or encamp on public property?",
        "Should the government provide subsidies for first-time homebuyers?",
        "Should the government increase funding for homeless shelters and services?",
        "Should new housing developments be required to include green spaces and parks?",
        "Should the government incentivize the construction of high density residential buildings?",
        "Should the government restrict the purchase of residential properties by foreign investors?"
    ],
    "Foreign Policy": [
        "Should the government increase or decrease military spending?",
        "Should the United States provide military supplies and funding to Ukraine?",
        "Should the U.S. remain in the United Nations?",
        "Should the U.S. remain in NATO?",
        "Should the U.S. continue to support Israel?",
        "Should the US increase or decrease foreign aid spending?",
        "Should the military be allowed to use enhanced interrogation techniques, such as waterboarding, to gain information from suspected terrorists?",
        "Should foreign terrorism suspects be given constitutional rights?",
        "Should the military fly drones over foreign countries to gain intelligence and kill suspected terrorists?",
        "Should every 18 year old citizen be required to provide at least one year of military service?",
        "Do you support the government calling for a permanent ceasefire in Gaza?",
        "Should Ukraine join NATO?"
    ]
}


def load_personas(filename, is_extended):
    with open(filename, 'r') as file:
        data = json.load(file)
    if is_extended:
        return data.get("extended_personas", {})
    else:
        return data.get("simple_personas", {})


def create_agents(personas, agent_names):
    agents = []
    for affiliation, configs in personas.items():
        if affiliation not in ["democrat2", "democrat3", "republican2", "republican3"]:
            agent_name = agent_names[affiliation]
            agents.append(DebateAgent(identifier=affiliation, name=agent_name, model=AGENT_MODEL, party=None, leaning=None, temperature=0.3))
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
        print(f"Agent: {agent.name}, Model: {agent.model}, Affiliation: {agent.identifier}\n")

        agent.generate_persona_prompt()

        # iterate over categories in questions
        for category, category_questions in questions.items():
            for question in category_questions:
                question_prompt = f"Answer the question from your political perspective in 50 words or less: {question}"
                response = agent.respond(debate_phase_prompt=None, conversation="", inst_prompt=question_prompt)

                print(f"Category: {category}")
                print(question)
                print(f"{agent.name}: {response}")
                
                csv_data.append([
                    agent.name,
                    agent.identifier.title(),
                    category,  # Add category here
                    question,
                    response
                ])
        
        if i % 10 == 0 or i == len(agents) - 1:  # save every 10 agents
            append_to_csv(csv_filename, csv_data)
            csv_data.clear()
            print(f"Processed and saved {i + 1} agents so far...")


def start_interivews(is_extended):
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    personas_file = os.path.join(BASE_PATH, "personas.json")
    if is_extended:
        csv_filename = os.path.join(BASE_PATH, "agent_validation_extended.csv")
    else:
        csv_filename = os.path.join(BASE_PATH, "agent_validation_simple.csv")

    csv_header = ["Agent Name", "Affiliation", "Category", "Question", "Response"]
    
    personas = load_personas(personas_file, is_extended)
    print(f"Using personas: {personas}\n")
    agents = create_agents(personas, AGENT_NAMES)
    create_csv(csv_filename, csv_header)
    interview_agents(agents, INTERVIEW_QUESTIONS, csv_filename)
    
    print(f"CSV file '{csv_filename}' has been created successfully.")



def evaluate_response(question, response, affiliation):
    political_affiliation = None

    if affiliation == "neutral":
        political_affiliation = "neutral American"
    elif affiliation == "republican":
        political_affiliation = "Republican American"
    elif affiliation == "democrat":
            political_affiliation = "Democrat American"

    print(f"Affiliation is: {affiliation}\n")

    prompt = f"""
    Does the response reflect the typical (or stereotypical) views of a {political_affiliation} on the question '{question}'? Answer "Yes" or "No".
    Explain your reasoning in less than 50 words.

    Response: {response}
    """

    try:
        if "mistral" in EVALUATION_MODEL:
            llm_response = ollama.chat(model=EVALUATION_MODEL, messages=[{"role": "user", "content": prompt}])
            llm_text = llm_response["message"]["content"]
        elif "gpt" in EVALUATION_MODEL:
            llm_response = client.chat.completions.create(
                model=EVALUATION_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )
            llm_text = llm_response.choices[0].message.content

        print(prompt)
        print("\n\n",llm_text)

        # extract answer: look for "Yes" or "No" at beginning or after a label
        answer_match = re.search(r'\b(?:Answer\s*:)?\s*(Yes|No)\b', llm_text, re.IGNORECASE)
        evaluation = answer_match.group(1).capitalize() if answer_match else "Error: Unable to parse"


        explanation_match = re.search(r'(?:Explanation\s*:)?\s*(.+)', llm_text, re.IGNORECASE | re.DOTALL)
        explanation = explanation_match.group(1).strip() if explanation_match else "Error: No explanation found"

        return evaluation, explanation
    except Exception as e:
        print("Exception: ", str(e))
        return "Error", str(e)


def evaluate_interviews(is_extended):
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    if is_extended:
        input_csv_filename = os.path.join(BASE_PATH, "agent_validation_extended.csv")
        output_csv_filename = os.path.join(BASE_PATH, "agent_validation_extended_evaluated_gpt.csv")
    else:
        input_csv_filename = os.path.join(BASE_PATH, "agent_validation_simple.csv")
        output_csv_filename = os.path.join(BASE_PATH, "agent_validation_simple_evaluated_gpt.csv")

    batch_size = 100

    with open(input_csv_filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

        print("READER FIELDNAMES = ", reader.fieldnames)

        fieldnames = reader.fieldnames + ["Evaluation", "Explanation"]

        for i, row in enumerate(rows):
            if "Evaluation" in row and row["Evaluation"]:  # skip already processed rows
                continue

            evaluation, explanation = evaluate_response(row.get("Question"), row.get("Response"), row["Affiliation"])
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
    # load both datasets if available
    if simple_csv is not None:
        df_simple = pd.read_csv(simple_csv)
        df_simple['Persona Type'] = 'Simple'
    else:
        df_simple = pd.DataFrame()

    if extended_csv is not None:
        df_extended = pd.read_csv(extended_csv)
        df_extended['Persona Type'] = 'Enhanced'
    else:
        df_extended = pd.DataFrame()
    
    # combine datasets
    df_combined = pd.concat([df_simple, df_extended])
    
    # convert evaluations to numerical scores
    df_combined['Score'] = df_combined['Evaluation'].apply(
        lambda x: 1 if str(x).lower() == 'yes' else 0
    )
    
    # map questions to their categories
    question_to_category = {}
    for category, questions in INTERVIEW_QUESTIONS.items():
        for q in questions:
            question_to_category[q] = category
    
    df_combined['Category'] = df_combined['Question'].map(question_to_category)
    
    # calculate mean alignment by category and persona type
    results = df_combined.groupby(['Persona Type', 'Category'])['Score'].mean().reset_index()
    
    return results

def create_radar_chart(data, output_file="radar_comparison.pdf"):
    categories = list(INTERVIEW_QUESTIONS.keys())
    N = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    # set up figure and axis
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # increase axis label font size
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([])
    for angle, label in zip(angles[:-1], categories):
        x = np.cos(angle) * 1.1  # push labels out slightly (1.0 is the default radius)
        y = np.sin(angle) * 1.1
        ha = 'center'
        if angle == 0 or angle == np.pi:
            ha = 'center'
        elif 0 < angle < np.pi:
            ha = 'left'
        else:
            ha = 'right'
        
        radius = 1.05
        if label == "Immigration":
            radius = 1.11

        ax.text(angle, radius, label, size=20, ha=ha, va='center')


    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1], ["25%", "50%", "75%", "100%"], color="grey", size=18)
    plt.ylim(0, 1)

    colors = ['#1f77b4', '#ff7f0e']
    for i, persona_type in enumerate(['Simple', 'Enhanced']):
        subset = data[data['Persona Type'] == persona_type]
        subset = subset.set_index('Category').reindex(categories).reset_index()
        values = subset['Score'].values.tolist()
        values += values[:1]
        
        ax.plot(angles, values, color=colors[i], linewidth=3, linestyle='solid',
                label=f'{persona_type} Personas')

    legend = ax.legend(loc='lower center',
                   bbox_to_anchor=(0.5, -0.16),  # adjust -0.15 as needed
                   fontsize=20, ncol=2, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(BASE_PATH, output_file), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart saved as {output_file}")


def visualise_evaluation(only_extended=False):
    simple_csv = os.path.join(BASE_PATH, "agent_validation_simple_evaluated_gpt.csv")
    extended_csv = os.path.join(BASE_PATH, "agent_validation_extended_evaluated_gpt.csv")
    
    if only_extended and not os.path.exists(extended_csv):
        raise FileNotFoundError(f"Extended personas CSV not found at {extended_csv}")
    
    if not only_extended:
        if not os.path.exists(simple_csv) or not os.path.exists(extended_csv):
            raise FileNotFoundError("Both simple and extended personas CSV files are required for comparison.")
    
    if only_extended:
        print("Only Extended persona data available. Visualising Extended persona evaluation.")
        prepared_data = load_and_prepare_data(None, extended_csv)
        create_radar_chart(prepared_data, output_file="extended_persona_comparison.pdf")
        
    else:
        print("Both Simple and Extended persona data available. Visualising both.")
        prepared_data = load_and_prepare_data(simple_csv, extended_csv)
        create_radar_chart(prepared_data)


if __name__ == "__main__":
    start_interivews(is_extended=False)
    # evaluate_interviews(is_extended=False)
    # visualise_evaluation(only_extended=False)

    # start_interivews(is_extended=True)
    # evaluate_interviews(is_extended=True)
    # visualise_evaluation(only_extended=True)
