This is a repo for the multiagent debate framework to improve the reasoning of agents in debates.

1) Install [Ollama](https://ollama.com/)
2) Run a model:
```
ollama run llama3.2:3b
```
3) Create a python virtual environment and install requirements
```
python3 -m venv venv
```
Activate the venv - csh:
```
source venv/bin/activate.csh
```
or with bash:
```
source venv/bin/activate
```

Install the python modules
```
pip install -r requirements.txt
```
4) Run the test ollama script and see that you get a result, you may need to update the selected model in the script
```
python3 test_ollama.py
```

## Running the debate:
To run the debate:
1) See the config in debate/debate_config.yaml, and update your settings as required. 
This includes changing the debate topics, enabling multiprocessing and more.

2) Run the debate with:
```
python3 debate/debate_runner.py
```


## Running the evaluation:
To run the evaluation:
1) Download the mistral model:
```
ollama pull mistral:7b
```
2) Run the evaluation script:
```
python3 evaluation/evaluation_runner.py
```

## Running the comparison

Once you have computed the evaluation on a set of debates while changing age, gender, model type, etc, you should now compare them to determine if there is a statistically significant difference between the results.

The comparison feature allows you to analyze whether different factors (e.g., gender, age) impact the debate outcomes using ANOVA and Levene’s test.

ANOVA tests whether the mean attitude scores differ significantly between group, while Levenes tests whether there is significantly different spread in results (1 group more volatile than the other)

In both cases, we look for p < 0.05 to for a statistically significant result.

To perform the compariso, we use the comparison directory

We take gender of the opinionated agents as an example.
1) Run all evaluations and set them aside - ensure that the eval_data transcripts for male and female are separate
2) In comparison/evaluated_data create a new directory. In this case we call it gender_opinionated
3) Create separate directories for however many categories you wish to compare. In our case, we create opinionated_female and opinionated_male
4) Into each directory paste in the neutral_republican_democrat directory from eval data - it must have already been evaluated and had the scores added 
5) Set the compare_path in comparison_config.yaml, and update any settings
6) Run comparison_runner.py
7) View the results in comparison_results.json
8) I have left gender opinionated in for reference - please ensure you copy its structure and it should work

## Running on Lab Machines
Installing ollama on the lab machines is a bit different. Please ask Fabian if you have any questions
You require a project directory, as the existing 10GB is not sufficient. You must email TSG for them to allocate you this. Again ask Fabian about this if you have questions. 

First navigate to your project directory

```
cd /cs/student/projects1/2021
cd <username>
```

Clone the repository to your project directory
Then run the lab_machine_install.sh script

```
bash lab_machine_install.sh
```

You should then add ollama to the path - this can vary by shell which you use - bash, csh, zsh. By default - the lab machines appear to use csh, so I reccomend that you add it first and foremost.

For Csh
```
echo 'set path = ("/cs/student/projects1/2021/$(whoami)/ollama/bin" $path)' >> ~/.cshrc
```

For bash
```
echo 'export PATH="/cs/student/projects1/2021/$(whoami)/ollama/bin:$PATH"' >> ~/.bashrc
```

For Zsh
```
echo 'export PATH="/cs/student/projects1/2021/$(whoami)/ollama/bin:$PATH"' >> ~/.zshrc
```


Then follow the instructions above, creating a venv, activating it, installing the required modules, and running the required models.

Then run test_ollama.py
```
python3 test_ollama.py
```

Finally, run the debate as before.
