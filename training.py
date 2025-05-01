import pandas as pd
import json
from tqdm import tqdm

#Load the training data from the file
train_data = pd.read_csv("data/train.csv")
# Drop useless columns from the data provided
train_df = train_data.drop(columns=["timestamp"])
del train_data #Clean memory

test_data = pd.read_csv("data/test.csv").drop(columns=["timestamp"])

with open("data/questions.json", 'r') as f:
    questions = json.load(f)

with open("data/concept.json", 'r') as f:
    concepts = json.load(f)

# Register a set of all users
all_users = set(train_df["uid"])

# Register a set of all concepts (keys of the 'concepts' dictionary)
all_concepts = list(concepts.keys())

# Create the dictionary with users as keys and concept dictionaries as values
user_dict = {user: {concept: 0 for concept in all_concepts} for user in all_users}

# Create a look up table that lists all the concepts a user has solved correctly
for _, training in tqdm(train_df.iterrows(), total=len(train_df), desc="Evaluating solved concepts"):
    # Some have several concepts that will then be together by _
    # Split them and treat as seperate instances
    split_concepts = training["concept_id"].split("_")
    for concept in split_concepts:
        user_dict[training["uid"]][concept] += training["response"] 

output_list = []
# Create an output guess by if the user has correctly solved such a concept before
for _, test in tqdm(test_data.iterrows(),total=len(test_data), desc="Creating Test file"):
    # Split the concepts in case there are several
    split_concepts = test["concept_id"].split("_")
    known_concepts = 0
    # Add the times the user correctly solved the concepts before
    for concept in split_concepts:
        known_concepts += user_dict[test["uid"]][concept]
    # Average out against the amount of different concepts present
    known_concepts /= len(split_concepts)

    # Aggregate the final submission
    output_list.append((test["uid"],known_concepts))

columns = ["uid","response"]
results = pd.DataFrame(output_list, columns=columns)
results.to_csv("output.csv", index=False)



print("Finished task, wrote results to output.csv")