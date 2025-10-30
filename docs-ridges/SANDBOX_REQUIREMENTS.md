# Sandbox requirements (summary)

Source: `evaluator/sandbox/sandbox_requirements.txt` from `ridgesai/ridges`.

This list defines the Python packages available inside the agent/evaluator sandbox. Your `agent.py` should assume these libraries exist and avoid adding new heavy deps at runtime. Prefer standard library when possible.

## Key categories
- Core ML/DL: torch, torchvision, torchaudio, tensorflow, keras
- NLP: nltk, spacy, gensim, transformers, sentence-transformers
- Tabular/metrics: numpy, pandas, scipy, scikit-learn, xgboost, lightgbm, catboost, statsmodels, seaborn, matplotlib
- Experimentation: mlflow, optuna, tqdm, joblib
- Web/API/runtime: fastapi (pinned 0.116.1), uvicorn (pinned 0.35.0), aiohttp
- Packaging/config: tox, toml, tomlkit
- Utilities: requests, urllib3, dill, tblib, appdirs, platformdirs, pickle-mixin
- Code quality: isort (<6), mccabe
- Data: datasets
- AutoGen: autogen-agentchat (0.7.4), autogen-ext[openai]
- Computer vision: opencv-python
- Pydantic: pydantic (>=2.5.0)
- Django-related utilities: asgiref (note: both ">=3.9.1" and "==3.10.0" are listed) and sqlparse (==0.5.3)

## Exact pins and potential conflicts
- Exact pins:
  - appdirs==1.4.4
  - asgiref==3.10.0 (also a ">=3.9.1" line above)
  - fastapi==0.116.1
  - sqlparse==0.5.3
  - uvicorn==0.35.0
  - autogen-agentchat==0.7.4
- Version ranges include many ">=" constraints which the sandbox resolver will satisfy.
- Duplicate/overlapping specs:
  - asgiref appears twice: ">=3.9.1" and "==3.10.0". Resolver should land on 3.10.0; treat it as pinned.
  - isort pinned as ">=4.2.5,<6" to avoid breaking changes in v6.
  - astroid range includes prerelease upper bound: ">=3.0.0a8,<=3.1.0-dev0".

## Implications for agent design
- You can safely import torch/transformers/sentence-transformers for embeddings or small models, but keep runtime short and memory bounded.
- Prefer algorithmic/deterministic solutions for Polyglot tasks; avoid large model downloads.
- For SWE-style patches, standard library + difflib should suffice; heavy ML libs are unnecessary.
- Networking is brokered by the inference gateway; use `requests` only to call the provided gateway endpoints.

## Best practices
- Determinism: fix seeds in numpy/torch when used.
- Performance: limit file reads/writes; avoid loading large models; cap prompt/context size.
- Compatibility: respect pinned FastAPI/uvicorn versions if building local tools.
- Security: don’t execute untrusted code; avoid shelling out.

## Full list (copied)
```
tox>=4.0.0
dill>=0.3.7
nltk>=3.8.0
toml>=0.7.1
spacy>=3.7.0
tblib>=3.1.0
torch>=2.1.0
tqdm>=4.65.0
gensim>=4.3.0
joblib>=1.3.0
keras>=2.15.0
mlflow>=2.8.0
numpy>=1.26.0
optuna>=3.4.0
pandas>=2.1.0
pytest>=7.4.0
scipy>=1.11.0
aiohttp>=3.9.0
appdirs==1.4.4
asgiref>=3.9.1
urllib3>=2.0.0
xgboost>=2.0.0
asgiref==3.10.0
catboost>=1.2.0
isort>=4.2.5,<6
lightgbm>=4.1.0
pydantic>=2.5.0
seaborn>=0.12.0
sqlparse==0.5.3
tomlkit>=0.10.1
uvicorn==0.35.0
datasets>=2.14.0
fastapi==0.116.1
mccabe>=0.6,<0.7
requests>=2.31.0
matplotlib>=3.7.0
torchaudio>=2.1.0
tensorflow>=2.15.0
autogen-ext[openai]
pickle-mixin>=1.0.2
platformdirs>=2.2.0
scikit-learn>=1.3.0
statsmodels>=0.14.0
torchvision>=0.16.0
opencv-python>=4.8.0
transformers>=4.35.0
autogen-agentchat==0.7.4
sentence-transformers>=2.2.0
astroid>=3.0.0a8,<=3.1.0-dev0
```
