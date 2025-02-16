# LLAMANG
Means: LLM LANG KAILANGAN

# Setup
1. Install Ollama from https://ollama.com/
2. Install models:
```shell 
ollama pull llama3.2:latest
ollama pull deepseek-r1:1.5b
```
3. Install python dependencies
```shell
py -m .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

# Usage
1. For running the model, change the model provided somewhere in llamang.py
2. Run script
```shell
py ./llamang.py
```
