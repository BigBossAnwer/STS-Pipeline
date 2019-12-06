# STS-Project

Semantic textual similarity project for Fall 2019 CS 6320 course at UT Dallas

## Prerequisites

* Python 3.X
* pip

## Getting Started

**Ensure you are using the pip associated with the desired venv**:

```bash
which pip
```

Which should point to the desired venv

Alternatively, invoke the desired venv directly:

```bash
/Path/To/desired_env/bin/python -m pip <commands>
```

Within the desired venv, install project requirements with:

```bash
pip install -U -r requirements.txt
```

Install the following NLTK WordNet data (again making sure to use the desired venv):

````bash
python -m nltk.downloader wordnet
python -m nltk.downloader omw
python -m nltk.downloader popular
````

Finally, before running any programs from ```sts_wrldom``` (or the notebooks), ensure that the ```PYTHONPATH``` system variable includes the ```STS-Project``` directory. For example:

```bash
echo $PYTHONPATH
/Path/to/Projects/STS-Project
```
