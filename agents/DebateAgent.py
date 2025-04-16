import os
import json
import ollama
import atexit
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

genai.configure(api_key=os.environ["GOOGLE_CLOUD_API_KEY"])

# optional cleanup for graceful shutdown
def cleanup():
    print("Shutting down cleanly...")

atexit.register(cleanup)


class DebateAgent:
    def __init__(self, identifier, name, party, leaning, model, temperature):
        self.identifier = identifier
        self.name = name
        self.persona_prompt = None
        self.party = party
        self.leaning = leaning
        self.model = model
        self.inst_prompt = None
        self.token_limit = 75,
        self.temperature = temperature


    def _load_extended_personas(self):
        persona_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extended_personas.json")
        try:
            with open(persona_file, "r", encoding="utf-8") as file:
                return json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading persona file: {e}")
            return {}


    def generate_persona_prompt(self, use_extended_persona=True):
        if use_extended_persona:
            extended_persona = self._load_extended_personas().get("generated_personas")
            self.persona_prompt =  extended_persona.get(self.identifier).get("baseline")
        else:
            party_support = f" who supports the {self.party} party" if self.party != None else ""
            self.persona_prompt = f"You are {self.name}, " \
                    f"{f'a {self.leaning}' if self.leaning else 'an'} American{party_support}."# \n"
                    # f"{self.debate_purpose}"


    def respond(self, debate_phase_prompt, conversation, inst_prompt):
        self.inst_prompt = inst_prompt  # for taxonomy generation, this is the prompt asking to generate one; for debates, it's asking agents to debate on a topic

        if "gemini" in self.model:
            model = genai.GenerativeModel(self.model)
            generation_config = GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=1024,
            )
            response = model.generate_content(
                f"{self.persona_prompt} {self.inst_prompt}\n{debate_phase_prompt if debate_phase_prompt is not None else ''}.\nConversation History: {conversation}",
                generation_config=generation_config
            )
            return response.text

        else:
            response = ollama.chat(
                model=self.model,
                options={"num_ctx": 4096, "temperature": self.temperature},
                messages=[{"role": "user", "content": f"{self.persona_prompt} {self.inst_prompt}\n{debate_phase_prompt if debate_phase_prompt != None else ''}. \nConversation History: {conversation}"}]
            )
            return response["message"]["content"]
