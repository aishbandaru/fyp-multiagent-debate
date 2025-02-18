import ollama


class DebateAgent:
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.prompt = "You are..."
        self.word_limit = 50

    def respond(self, conversation_history):
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": f"{self.prompt}."}]
        )
        return response["message"]["content"]
