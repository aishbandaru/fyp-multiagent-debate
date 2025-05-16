import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[{"role": "user", "content": "Write a Python function to calculate Fibonacci numbers."}]
)

print(response["message"]["content"])
