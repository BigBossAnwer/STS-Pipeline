import argparse
from pathlib import Path

import spacy
from spacy import displacy

from sts_wrldom.corpusReader import read_data
from sts_wrldom.utils import log_frame


def preprocess(df, name=None, tag=None, log=False):
    """Enriches a dataframe (inplace) such that str columns: [s1, s2] have 
    their associated tokens, lemmas, POS tags and spaCy docs concatenated in as columns 
    with names:
    s1_tokens, s1_lemmas, s1_pos, s1_docs (likewise for s2)
    
    Args:
        df: the dataframe to be inplace enriched 
        name (str, optional): the name for the dataframe .csv file. Defaults to None.
        tag (str, optional): the tag for the logged dataframe .csv file . Defaults to None.
        log (bool, optional): specifies logging of the enriched dataframe.
            Defaults to False.
    """
    nlp = spacy.load("en_core_web_sm")

    if name is None:
        announce = "Enriching dataframe with tokens, lemmas, and POS tags..."
    else:
        announce = f"Enriching {name} dataframe with tokens, lemmas, and POS tags..."
    print(announce)
    parse_cols = ["s1", "s2"]
    for col in parse_cols:
        tokens = []
        lemmas = []
        pos = []
        docs = []
        parse_fail = 0

        for doc in nlp.pipe(df[col].values, batch_size=50, n_threads=4):
            if doc.is_tagged:
                tokens.append([n.text for n in doc])
                lemmas.append([n.lemma_ for n in doc])
                pos.append([n.pos_ for n in doc])
                docs.append(doc)
            else:
                # Ensure parse lists have the same number of entries as the original
                #   Dataframe regardless of parse failure
                parse_fail += 1
                tokens.append(None)
                lemmas.append(None)
                pos.append(None)
                docs.append(None)

        print(f"{col.upper()} parse failures: {parse_fail}")
        df[col + "_tokens"] = tokens
        df[col + "_lemmas"] = lemmas
        df[col + "_pos"] = pos
        df[col + "_docs"] = docs

    if log:
        if name is None or tag is None:
            print("Error: Logging requires a name and a tag")
        else:
            log_frame(df.drop(columns=["s1_docs", "s2_docs"]), name=name, tag=tag)
    print()


def preprocess_raw(df):
    """Returns spaCy processed documents as a list of spaCy doc tuples. 
    Saves multiple passes through the spaCy NLP pipe in sts_wrldom architecture.
    
    Args:
        df: the source dataframe with columns: [s1, s2]
    
    Returns:
        list(tuple(spaCy_doc, spaCy_doc)): a list of 2-tuples where every tuple in the 
        list corresponds to a row in the dataframe, and tuple[0] corresponds to s1, 
        tuple[1] to s2.
    """
    nlp = spacy.load("en_core_web_sm")

    print("Enriching data from dataframe...")

    parse_cols = ["s1", "s2"]
    s1_docs = []
    s2_docs = []
    for col in parse_cols:
        parse_fail = 0

        for doc in nlp.pipe(df[col].values, batch_size=50, n_threads=4):
            if doc.is_parsed:
                if col == "s1":
                    s1_docs.append(doc)
                else:
                    s2_docs.append(doc)
            else:
                # Ensure parse lists have the same number of entries as the original
                #   Dataframe regardless of parse failure
                parse_fail += 1
                if col == "s1":
                    s1_docs.append(None)
                else:
                    s2_docs.append(None)

        print(f"{col.upper()} parse failures: {parse_fail}")

    print()

    return list(zip(s1_docs, s2_docs))


def main():
    description = (
        f"Pipeline for NLP enrichment as specified by the dataframe structure found "
        f"in {Path('src/corpusReader.py')}. Running this standalone amounts to testing"
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--corpus_path",
        help=f"{Path('Path/To/Corpus/*-set.txt')}, Default: {Path('data/*-set.txt')}",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help=f"Suppresses logging of produced log files to: {Path('log/*')}",
        action="store_true",
    )
    args = parser.parse_args()
    log = not args.quiet

    dfs = read_data(["dev", "train", "test"], args.corpus_path, log)

    for frame in dfs.keys():
        preprocess(dfs[frame], name=frame, tag="enriched", log=log)

    s1_docs = []
    s2_docs = []
    for frame in dfs.keys():
        print("Enriched " + frame + " head: ")
        print(dfs[frame].head(5), "\n")
        tmp_list = list(dfs[frame]["s1_docs"])
        s1_docs += tmp_list[:3]
        s1_docs += tmp_list[-3:]
        tmp_list = list(dfs[frame]["s2_docs"])
        s2_docs += tmp_list[:3]
        s2_docs += tmp_list[-3:]

    # Interleaves lists s1 and s2: [s1[0], s2[0], s1[1], s2[1]...]
    docs = [val for pair in zip(s1_docs, s2_docs) for val in pair]
    if log:
        print("Logging WordNet Feature Extraction Sample...")
        Path("log").mkdir(exist_ok=True)
        log_path = Path("log", "Task2_WordNetFeatures_Sample.txt")

        with log_path.open("w", encoding="utf-8") as out_fh:
            from pywsd import disambiguate

            for doc in docs:
                out_fh.writelines(doc.text)
                disam = disambiguate(doc.text)
                for word, syn in disam:
                    if syn:
                        out_fh.writelines(f"\n{word} | {syn} | {syn.definition()}")
                        out_fh.writelines(f"\tHypernyms:\n\t{syn.hypernyms()}")
                        out_fh.writelines(f"\tHyponyms:\n\t{syn.hyponyms()}")
                        mero = (
                            f"\tMeronyms:\n\t{syn.part_meronyms()}\n"
                            f"\t{syn.substance_meronyms()}"
                        )
                        out_fh.writelines(mero)
                        holo = (
                            f"\tHolonyms:\n\t{syn.part_holonyms()}\n"
                            f"\t{syn.substance_holonyms()}"
                        )
                        out_fh.writelines(holo)
                    else:
                        out_fh.writelines(f"\n{word}")
                out_fh.writelines("\n\n--------\n\n")

    options = {"compact": True, "color": "blue"}
    # Save to .svg
    if log:
        svg = displacy.render(docs, style="dep", options=options)
        log_path = Path("log", "depTrees.svg")
        with log_path.open("w", encoding="utf-8") as dep_log:
            dep_log.write(svg)

    # Dependency Parse Tree Visualization LocalHost Serve:
    displacy.serve(docs, style="dep", options=options)


if __name__ == "__main__":
    main()
