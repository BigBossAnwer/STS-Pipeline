import spacy
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

from spacy_wordnet.wordnet_annotator import WordnetAnnotator

# Load an spacy model (supported models are "es" and "en")
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(WordnetAnnotator(nlp.lang), after="tagger")

# Imagine we want to enrich the following sentence with synonyms
sentence = nlp("I want to withdraw 5,000 euros")

# spaCy WordNet lets you find synonyms by domain of interest
# for example economy
economy_domains = ["finance", "banking"]
enriched_sentence = []

# For each token in the sentence
for token in sentence:
    # We get those synsets within the desired domains
    synsets = token._.wordnet.wordnet_synsets_for_domain(economy_domains)
    if synsets:
        lemmas_for_synset = []
        for s in synsets:
            # If we found a synset in the economy domains
            # we get the variants and add them to the enriched sentence
            lemmas_for_synset.extend(s.lemma_names())
            enriched_sentence.append("({})".format("|".join(set(lemmas_for_synset))))
    else:
        enriched_sentence.append(token.text)

# Let's see our enriched sentence
print(" ".join(enriched_sentence))

# Look upon my hack and weep, for i am wrath, destroyer of modularity
# ripped from spacy-wordnet internals
from nltk.corpus.reader.wordnet import (
    ADJ as WN_ADJ,
    ADV as WN_ADV,
    NOUN as WN_NOUN,
    VERB as WN_VERB,
    Synset,
)

from spacy.parts_of_speech import ADJ, ADV, NOUN, VERB, AUX


__WN_POS_MAPPING = {ADJ: WN_ADJ, NOUN: WN_NOUN, ADV: WN_ADV, VERB: WN_VERB, AUX: WN_VERB}


def spacy2wordnet_pos(spacy_pos: int):
    return __WN_POS_MAPPING.get(spacy_pos)


token = nlp("fly")[0]
t = wn.synsets(token.text, spacy2wordnet_pos(token.pos))[0]
s = wn.synsets("walk", wn.VERB)[0]
brown_ic = wordnet_ic.ic("ic-brown.dat")
print(s.lin_similarity(t, brown_ic))


# Similarity score as described in Pawar, 2018
import numpy as np

s1 = wn.synset("car.n.01")
s2 = wn.synset("motorcycle.n.01")
subsumer = s1.lowest_common_hypernyms(s2, simulate_root=True)[0]
h = subsumer.max_depth() + 1
syn1_dist_subsumer = s1.shortest_path_distance(subsumer, simulate_root=True)
syn2_dist_subsumer = s2.shortest_path_distance(subsumer, simulate_root=True)
## same thing as two prior lines:
# l = s1.shortest_path_distance(s2)
l = syn1_dist_subsumer + syn2_dist_subsumer
print(f"subsumer = {subsumer}")
print(f"l = {l}")
print(f"h = {h}")
alpha = 0.1
beta = 0.45
f = np.exp(-alpha * l)
g1 = np.exp(beta * h)
g2 = np.exp(-beta * h)
g = (g1 - g2) / (g1 + g2)
sim = f * g
print(f"f = {f}")
print(f"g = {g}")
print(f"Sim = {sim}")
