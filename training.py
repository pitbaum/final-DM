import pandas as pd
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
import os
import torch.nn.functional as F

"""
    Load data
"""

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

"""
    Create question embeddings
"""

if not os.path.exists(os.getcwd()+"/question_embeddings.pt"):

    print("question embeddings dont exist yet, creating them")
    # Load the sentence transformer model
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model.eval()
    questions_list = []
    for i in tqdm(range(len(questions))):
        question = questions[str(i)]
        content = question["content"]
        kc_routes = ""
        for x in question["kc_routes"]:
            kc_routes += x
        answer = ""
        for x in question["answer"]:
            answer += x  # Fixed: use correct field instead of repeating kc_routes
        analysis = question["analysis"]
        q_type = question["type"]
        # Combine all fields into a single string
        questions_list.append(content + answer + analysis + q_type)

    batch_size = 32
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(questions_list), batch_size), desc="Encoding questions"):
            combined_batch = questions_list[i:i+batch_size]

            # Encode the batch
            batch_embeddings = model.encode(
                combined_batch,
                batch_size=batch_size,
                convert_to_tensor=True,
            )
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    question_embeddings = torch.cat(embeddings, dim=0)
    print("Finished creating question embeddings")

    torch.save(question_embeddings, "question_embeddings.pt")
    print("Saved to file")

else:
    question_embeddings = torch.load("question_embeddings.pt")

"""
    Generate user known concept dict
"""
"""
# Create the dictionary with users as keys and concept dictionaries as values
user_dict = {user: {concept: 0 for concept in all_concepts} for user in all_users}

# Create a look up table that lists all the concepts a user has solved correctly
for _, training in tqdm(train_df.iterrows(), total=len(train_df), desc="Evaluating solved concepts"):
    # Some have several concepts that will then be together by _
    # Split them and treat as seperate instances
    split_concepts = training["concept_id"].split("_")
    for concept in split_concepts:
        if training["response"] == 0:
            score = -1
        else:
            score = 1
        user_dict[training["uid"]][concept] += score 
"""
"""
    Generate a user question embedding
"""

if not os.path.exists(os.getcwd() + "/known_user_embeddings.pt"):
    # Create a user_id key, list of known question ids dict
    # And one for questions that were anwsered wrongly
    user_known_question_dict = {user: [] for user in all_users}
    user_unknown_question_dict = {user: [] for user in all_users}
    for _, training_data in tqdm(train_df.iterrows(), total=len(train_df), desc="Create user embeddings"):
        # Append the question to the known dict if the response was correct
        if training_data["response"] == 1:
            user_known_question_dict[training_data["uid"]].append(training_data["question_id"])
        else:
            user_unknown_question_dict[training_data["uid"]].append(training_data["question_id"])

    torch.save(user_known_question_dict, "known_user_embeddings.pt")

    torch.save(user_unknown_question_dict, "unknown_user_embeddings.pt")

    # Create user embeddings tuple list (user id, user embedding)
    user_known_embeddings = {user: [] for user in all_users}
    for uid in user_known_question_dict.keys():
        looked_up_question_embeddings = []
        for question_id in user_known_question_dict[uid]:
            looked_up_question_embeddings.append(question_embeddings[question_id])
        if looked_up_question_embeddings == []:
            user_embedding = torch.stack([torch.tensor(0.0) for _ in range(384)])
        else:
            user_embedding = torch.mean(torch.stack(looked_up_question_embeddings),dim=0)
        user_known_embeddings[uid] = user_embedding

    torch.save(user_known_embeddings, "known_user_embeddings.pt")

    # Create user embedding for questions that were wrongly anwsered
    user_unknown_embeddings = {user: [] for user in all_users}
    for uid in user_unknown_question_dict.keys():
        looked_up_question_embeddings = []
        for question_id in user_unknown_question_dict[uid]:
            looked_up_question_embeddings.append(question_embeddings[question_id])
        if looked_up_question_embeddings == []:
            user_embedding = torch.stack([torch.tensor(0.0) for _ in range(384)])
        else:
            user_embedding = torch.mean(torch.stack(looked_up_question_embeddings),dim=0)
        user_unknown_embeddings[uid] = user_embedding

    torch.save(user_unknown_embeddings, "unknown_user_embeddings.pt")

else:
    user_known_embeddings = torch.load("known_user_embeddings.pt")
    user_unknown_embeddings = torch.load("unknown_user_embeddings.pt")

# Get the similarity of the test question and the known user embedding and the unknown user embedding
# The final result will then be (known similarity - unknown similarity)
final_result = []
for _,data in test_data.iterrows():
    user_id = data["uid"]
    known_similarity = torch.cosine_similarity(F.normalize(user_known_embeddings[user_id], p=2, dim=0),F.normalize(question_embeddings[data["question_id"]], p=2, dim=0), dim=0)
    unknown_similarity = torch.cosine_similarity(F.normalize(user_unknown_embeddings[user_id],p=2, dim=0), F.normalize(question_embeddings[data["question_id"]], p=2, dim=0), dim=0)
    final_result.append((user_id, (known_similarity-unknown_similarity).item()))

"""
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
"""
    
columns = ["uid","response"]
results = pd.DataFrame(final_result, columns=columns)
results.to_csv("output.csv", index=False)

print("Finished task, wrote results to output.csv")