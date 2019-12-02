# STS-Project

Semantic textual similarity project for Fall 2019 CS 6320 course at UT Dallas

## Prerequisites

* Python 3.X
* pip

**Ensure you are using the pip associated with the desired venv**:

```bash
which pip
```

Which should point to the desired venv

Alternatively, envoke the desired venv directly:

```bash
/Path/To/desired_env/bin/python -m pip <commands>
```

Within the desired venv, install project requirements with:

```bash
pip install -U -r requirements.txt
```

**To work with / run depTFIDFModel or depTFIDFModelTest**:

```bash
pip install -U -r requirements.large.txt
```

Finally, install the following NLTK wordnet data (again making sure to use the desired venv):

````bash
python -m nltk.downloader wordnet
python -m nltk.downloader omw
````
