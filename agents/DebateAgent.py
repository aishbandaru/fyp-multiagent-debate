import ollama


class DebateAgent:
    def __init__(self, name, model, inst_prompt):
        self.name = name
        self.model = model
        self.inst_prompt = inst_prompt  #"You are..."
        self.word_limit = 50

    def respond(self, conversation_history):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": f"{self.inst_prompt}. Conversation History: {conversation_history}"}]
        )
        return response["message"]["content"]
