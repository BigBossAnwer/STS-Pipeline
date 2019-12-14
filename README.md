# STS-Project

Semantic Textual Similarity project for CS 6320 - Natural Language Processing - Fall 2019, UT Dallas

## Prerequisites

* Python 3.X
* pip

## Getting Started

### Ensure you are using the pip associated with the desired Python env:

```bash
which pip
> /Path/to/desired_venv/bin/pip
```

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

### Before running any programs / notebooks dependent on Pywsd, patch Pywsd 1.2.4:

Programs dependent on Pywsd:

* ensembleModels.py
* pawarModel.py

Notebooks dependent on Pywsd:

* ensembleModels-Dev-Train.ipynb
* ensembleModels-Test.ipynb
* pawarModel-Dev-Train.ipynb
* pawarModel-Test.ipynb

As of Pywsd 1.2.4 (future releases may render this patch obsolete), a bug exists in Pywsd that will cause word sense disambiguation using ```pywsd.max_similarity``` to fail with an IndexError.
To patch this, find the Pywsd module in the site packages of the env that it was installed in / the env that the STS-Project will be run within:

Example location:

```bash
/Path/to/desired_venv/lib/python3.8/site-packages/pywsd/
```

Within the method ```max_similarity``` in ```.../site-packages/pywsd/similarity.py``` add the following before the return statement:

```python
    if not len(result):
        return None
```

```max_similarity``` should now look like:

```python
def max_similarity(context_sentence: str, ambiguous_word: str, option="path",
                   lemma=True, context_is_lemmatized=False, pos=None, best=True) -> "wn.Synset":
    """
    Perform WSD by maximizing the sum of maximum similarity between possible
    synsets of all words in the context sentence and the possible synsets of the
    ambiguous words (see https://ibin.co/4gG9zUlejUUA.png):
    {argmax}_{synset(a)}(\sum_{i}^{n}{{max}_{synset(i)}(sim(i,a))}

    :param context_sentence: String, a sentence.
    :param ambiguous_word: String, a single word.
    :return: If best, returns only the best Synset, else returns a dict.
    """
    ambiguous_word = lemmatize(ambiguous_word)
    # If ambiguous word not in WordNet return None
    if not wn.synsets(ambiguous_word):
        return None
    if context_is_lemmatized:
        context_sentence = word_tokenize(context_sentence)
    else:
        context_sentence = [lemmatize(w) for w in word_tokenize(context_sentence)]
    result = {}
    for i in wn.synsets(ambiguous_word, pos=pos):
        result[i] = 0
        for j in context_sentence:
            _result = [0]
            for k in wn.synsets(j):
                _result.append(sim(i,k,option))
            result[i] += max(_result)

    if option in ["res","resnik"]: # lower score = more similar
        result = sorted([(v,k) for k,v in result.items()])
    else: # higher score = more similar
        result = sorted([(v,k) for k,v in result.items()],reverse=True)
    
    if not len(result):
        return None
    
    return result[0][1] if best else result
```

Save the edited file. All STS-Project programs / notebooks dependent on Pywsd should now run without issue.

### Before running programs / notebooks, ensure environment variables are correct:

Verify that the ```PYTHONPATH``` system variable includes the ```sts_wrldom``` directory. For example:

```bash
echo $PYTHONPATH
> /Path/to/Projects/STS-Project/sts_wrldom
```

Ensure that the working directory of the terminal the program will run in is the STS-Project root directory. For example:

```bash
pwd
> /Path/to/Projects/STS-Project
```

## Usage

Standalone programs are:

* corpusReader
* enrichPipe
* depTFIDFModel
* pawarModel
* ensembleModels

All programs can run with no command line options, however all programs offer command line options. For available options, use:

```bash
python <program>.py -h
```

## Warning:

```STS-Project/notebooks/embedModel-Dev-Train-Test.ipynb``` was built and run directly inside Google Colabs. Avoid running it in a local environment as it has some ```pip3``` installs that might clutter the local environment

## References:

Pawar, Atish, and Vijay Mago. "Calculating the similarity between words and sentences using a lexical database and corpus statistics." arXiv preprint arXiv:1802.05667 (2018).

Cer, Daniel, Yinfei Yang, Sheng-yi Kong, Nan Hua, Nicole Limtiaco, Rhomni St John, Noah Constant et al. "Universal sentence encoder." arXiv preprint arXiv:1803.11175 (2018).
