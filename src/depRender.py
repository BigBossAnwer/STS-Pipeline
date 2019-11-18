from pathlib import Path

import pandas as pd
import spacy
from spacy import displacy

from corpusReader import read_data
from enrichPipe import preprocess

nlp = spacy.load("en_core_web_sm")
dev = read_data(["dev"])
dev = preprocess(dev, name="dev", tag="enriched", log=True)

docs = []
cols = ["s1", "s2"]
# First 5 rows
for i in range(5):
    for col in cols:
        doc = nlp(dev[col].iloc[i])
        docs.append(doc)

# Last 5 rows
for i in range(dev.shape[0] - 5, dev.shape[0]):
    for col in cols:
        doc = nlp(dev[col].iloc[i])
        docs.append(doc)

# LocalHost Serve:
displacy.serve(docs, style="dep")

# # Save to .svg
# svg = displacy.render(docs, style="dep")
# log_path = Path("log", "dep.svg")
# with log_path.open("w", encoding="utf-8") as dep_log:
#     dep_log.write(svg)
