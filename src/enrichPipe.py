import pandas
import spacy

from corpusReader import log_frame, read_data

nlp = spacy.load("en_core_web_sm")

dev = read_data(["dev"])
parse_cols = ["s1", "s2"]
for col in parse_cols:
    tokens = []
    lemma = []
    pos = []
    parse_fail = 0

    for doc in nlp.pipe(dev[col].values, batch_size=50, n_threads=4):
        if doc.is_parsed:
            tokens.append([n.text for n in doc])
            lemma.append([n.lemma_ for n in doc])
            pos.append([n.pos_ for n in doc])
        else:
            # Ensure parse lists have the same number of entries as the original Dataframe
            #   regardless of parse failure
            parse_fail += 1
            tokens.append(None)
            lemma.append(None)
            pos.append(None)

    print(f"{col.upper()} Failues: {parse_fail}")
    dev[col + "_tokens"] = tokens
    dev[col + "_lemma"] = lemma
    dev[col + "_pos"] = pos

log_frame(dev, name="dev", tag="enriched")
