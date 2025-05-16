# Multiagent Debate Framework

This repository provides a framework for conducting multiagent debates aimed at improving agent reasoning through structured argumentation and evaluation.

## Getting Started

You can run this framework using either:

- **Local models** (via [Ollama](https://ollama.com)), or  
- **Cloud APIs** (e.g., Gemini and OpenAI)

### Option 1: Local Models via Ollama

1. **Install Ollama**  
   Download and install from [https://ollama.com](https://ollama.com)

2. **Run a Model**  
   To download and test a model:
   ```bash
   ollama run llama3.2:3b
   ```

## Option 2: Cloud APIs (Gemini / OpenAI)

If you prefer to use models via APIs, ensure you have the correct keys. Add the following to your environment:

```bash
export OPENAI_API_KEY=your_openai_key
export GEMINI_API_KEY=your_gemini_key
```

You can add these lines to your .bashrc, .zshrc, or another startup script to persist them between sessions.

## Setup Instructions

1. Create a Python Virtual Environment

```bash
python3 -m venv venv
```

2. Activate the Virtual Environment

For Bash:

```bash
source venv/bin/activate
```

For C Shell:

```bash
source venv/bin/activate.csh
```

3. Install Required Python Modules

```bash
pip install -r requirements.txt
```

4. Test Ollama Integration

Run the example script to verify everything is working. You may need to update the selected model in the script.

```bash
python3 test_ollama.py
```

## Running the Taxonomy Generation
1. Configure Settings
    Modify the following parameters in debate/debate_config.yaml as needed:
    - Number of taxonomy rounds
    - Number of taxonomy iterations
    - Taxonomy tree depth
    - The LLM agents involved

2. Start Taxonomy Generation
    ```bash
    python3 debate/taxonomy_runner.py
    ```

## Running the Debate
1. Update Configuration
    Edit debate/debate_config.yaml to specify:
    - Number of debate rounds and iterations
    - Debate topics
    - Debate structures
    - Participating LLM agents

2. Run the Debate
    ```bash
    python3 debate/debate_runner.py
    ```

## Running Evaluations

### Taxonomy Evaluation

1. Update Evaluation Configuration
    In evaluation/eval_config.yaml, select the appropriate debate topics and structure settings.

2. Run the Taxonomy Evaluation
    ```bash
    python3 debate/taxonomy_evaluator.py
    ```

### Debate Evaluation
1. Update Evaluation Configuration
    Similarly, update evaluation/eval_config.yaml with the required evaluation parameters.

2. Run the Debate Evaluation
    ```bash
    python3 evaluation/evaluation_runner.py
    ```
