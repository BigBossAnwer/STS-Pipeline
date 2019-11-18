from corpusReader import read_data
from enrichPipe import preprocess

# Usage:
# preprocess(
#     df=df, # required, the dataframe to enrich with POS, lemmas, and tokens
#     name="some-name", # optional, df name for logging purposes, required if log=True
#     tag="some-tag", # optional, df tag for logging purposes, required if log=True
#     log=True, # optional, logs enriched dataframe to log/name_tag.csv
# )
# df passed in gets modified in place, returns the enriched dataframe
dev = read_data(which_sets=["dev"])
preprocess(dev)
# equivalently
# dev = process(dev)

print(dev.head())
