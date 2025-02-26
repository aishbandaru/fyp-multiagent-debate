import ollama


class DebateAgent:
    def __init__(self, name, model, inst_prompt, stance=None):
        self.name = name
        self.model = model
        self.inst_prompt = inst_prompt
        self.stance = stance
        self.word_limit = 50

    def respond(self, debate_phase_prompt, conversation):
        if self.stance == "proponent":
            stance_prompt = "You are a Democrat."
        elif self.stance == "opponent":
            stance_prompt = "You are a Republican."
        else:
            stance_prompt = ""

        response = ollama.chat(
            model=self.model,
            # options={"num_ctx": 4096, "temperature": 0.1},
            messages=[{"role": "user", "content": f"{stance_prompt} {self.inst_prompt}\n{debate_phase_prompt if debate_phase_prompt != None else ''} \nConversation History: {conversation}"}]
        )

        return response["message"]["content"]
