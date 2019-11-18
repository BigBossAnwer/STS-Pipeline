from corpusReader import read_data

# Usage: 
# read_data(
#     the_sets_you_want=["dev", "test", "train"], # required
#     custom_path_to_data=path # optional
#     log=True/False # optional, logs dataframes to log/*.csv
#     )
frames = read_data(["train", "test"])
dev = read_data(which_sets=["dev"]) # another access pattern
train, test = frames["train"], frames["test"]

print(train.head())
print(test.head())
print(dev.head())
