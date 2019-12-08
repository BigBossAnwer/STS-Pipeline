# STS-Project

Semantic Textual Similarity project for CS 6320 - Natural Language Processing - Fall 2019, UT Dallas

## Prerequisites

* Python 3.X
* pip

## Getting Started

### Ensure you are using the pip associated with the desired Python env:

```bash
which pip
/Path/to/desired_venv/bin/pip
```

Which should point to the desired env

Alternatively, invoke the desired env directly:

```bash
/Path/To/desired_venv/bin/python -m pip <commands>
```

### Within the desired env, install project requirements with:

```bash
pip install -U -r requirements.txt
```

Install the following NLTK WordNet data (again making sure to use the desired env):

````bash
python -m nltk.downloader wordnet
python -m nltk.downloader omw
python -m nltk.downloader popular
````

### Before running programs / notebooks, ensure environment variables are correct:

Verify that the ```PYTHONPATH``` system variable includes the ```sts_wrldom``` directory. For example:

```bash
echo $PYTHONPATH
/Path/to/Projects/STS-Project/sts_wrldom
```

Ensure that the working directory of the terminal the program will run in is the STS-Project root directory. For example:

```bash
pwd
/Path/to/Projects/STS-Project
```

## Usage

Standalone programs are:

* corpusReader
* enrichPipe
* depTFIDFModel
* pawarModel
* ensembleModels

All programs can run with no command line options, however all programs offer command line options. Use:

```bash
python <program>.py -h
```

for available options.

## Warning:

```STS-Project/notebooks/embedModel-Dev-Train-Test.ipynb``` was built and run directly inside Google Colabs. Avoid running it in a local environment as it has some ```pip3``` installs that might clutter the local environment

## References:

Pawar, Atish, and Vijay Mago. "Calculating the similarity between words and sentences using a lexical database and corpus statistics." arXiv preprint arXiv:1802.05667 (2018).

Cer, Daniel, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St John, Noah Constant et al. "Universal sentence encoder." arXiv preprint arXiv:1803.11175 (2018).
