import spacy

from spacy_wordnet.wordnet_annotator import WordnetAnnotator 

# Load an spacy model (supported models are "es" and "en") 
nlp = spacy.load('en_core_web_sm')
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

# Imagine we want to enrich the following sentence with synonyms
sentence = nlp('I want to withdraw 5,000 euros')

# spaCy WordNet lets you find synonyms by domain of interest
# for example economy
economy_domains = ['finance', 'banking']
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
            enriched_sentence.append('({})'.format('|'.join(set(lemmas_for_synset))))
    else:
        enriched_sentence.append(token.text)

# Let's see our enriched sentence
print(' '.join(enriched_sentence))