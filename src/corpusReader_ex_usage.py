from corpusReader import read_data

# Usage:
# read_data(
#     the_sets_you_want=["dev", "test", "train"], # required
#     custom_path_to_data=path # optional
#     log=True/False # optional, logs dataframes to log/*.csv
#     )
# returns either a single dataframe, or a dictionary of all requested dataframes
frames = read_data(["train", "test"])  # dictionary of dfs
train, test = frames["train"], frames["test"]
dev = read_data(which_sets=["dev"])  # another access pattern, single df

print(train.head())
print(test.head())
print(dev.head())
