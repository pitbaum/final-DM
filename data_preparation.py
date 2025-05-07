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
    
    model = SentenceTransformer('shibing624/text2vec-base-chinese') # Make sure to use a model that knows chinese
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
    Generate a user question embedding
"""

if (not os.path.exists(os.getcwd() + "/known_user_question_embeddings.pt")) or (not os.path.exists(os.getcwd() + "/unknown_user_question_embeddings.pt")):
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

    # Create user embeddings tuple list (user id, user embedding)
    user_known_embeddings = {user: [] for user in all_users}
    for uid in user_known_question_dict.keys():
        looked_up_question_embeddings = []
        for question_id in user_known_question_dict[uid]:
            looked_up_question_embeddings.append(question_embeddings[question_id])
        if looked_up_question_embeddings == []:
            user_embedding = torch.stack([torch.tensor(0.0) for _ in range(768)])
        else:
            user_embedding = torch.mean(torch.stack(looked_up_question_embeddings),dim=0)
        user_known_embeddings[uid] = user_embedding

    torch.save(user_known_embeddings, "known_user_question_embeddings.pt")

    # Create user embedding for questions that were wrongly anwsered
    user_unknown_embeddings = {user: [] for user in all_users}
    for uid in user_unknown_question_dict.keys():
        looked_up_question_embeddings = []
        for question_id in user_unknown_question_dict[uid]:
            looked_up_question_embeddings.append(question_embeddings[question_id])
        if looked_up_question_embeddings == []:
            user_embedding = torch.stack([torch.tensor(0.0) for _ in range(768)])
        else:
            user_embedding = torch.mean(torch.stack(looked_up_question_embeddings),dim=0)
        user_unknown_embeddings[uid] = user_embedding

    torch.save(user_unknown_embeddings, "unknown_user_question_embeddings.pt")

else:
    user_known_embeddings = torch.load("known_user_question_embeddings.pt")
    user_unknown_embeddings = torch.load("unknown_user_question_embeddings.pt")

"""
    Generate concept embeddings 
"""

if not os.path.exists(os.getcwd()+"/concept_embeddings.pt"):
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    model.eval()
    concept_list = []
    for i in tqdm(range(len(concepts))):
        concept = concepts[str(i)]
        concept_list.append(concept)

    batch_size = 32
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(concept_list), batch_size), desc="Encoding concepts"):
            combined_batch = concept_list[i:i+batch_size]

            # Encode the batch
            batch_embeddings = model.encode(
                combined_batch,
                batch_size=batch_size,
                convert_to_tensor=True,
            )
            embeddings.append(batch_embeddings.cpu())

    # Concatenate all embeddings
    concept_embeddings = torch.cat(embeddings, dim=0)
    print("Finished creating question embeddings")

    torch.save(concept_embeddings, "concept_embeddings.pt")
    print("Saved to file")

else:
    concept_embeddings = torch.load("concept_embeddings.pt")
    print("Read the concept embeddings from file")

"""
    Generate the user concept embeddings
"""

if (not os.path.exists(os.getcwd() + "/known_user_concept_embeddings.pt")) or (not os.path.exists(os.getcwd() + "/unknown_user_concept_embeddings.pt")):
    # Create a user_id key, list of known concpet ids dict
    # And One for concepts that were anwsered wrongly
    user_known_concept_dict = {user: [] for user in all_users}
    user_unknown_concept_dict = {user: [] for user in all_users}
    for _, train_data in tqdm(train_df.iterrows(), total=len(train_df), desc="Create user concept"):
        if train_data["response"] == 1:
            user_known_concept_dict[train_data["uid"]].append(train_data["concept_id"])
        else:
            user_unknown_concept_dict[train_data["uid"]].append(train_data["concept_id"])
    # Create user embeddings tuple list (user id, user concept embedding)
    user_known_concept_embeddings = {user: [] for user in all_users}
    for uid in user_known_concept_dict.keys():
        looked_up_concept_embeddings = []
        for concept_id in user_known_concept_dict[uid]:
            # split concept id by _ since sometimes several are registered at once
            concept_ids = concept_id.split("_")
            for x in concept_ids:
                looked_up_concept_embeddings.append(concept_embeddings[int(x)])
        if looked_up_concept_embeddings == []:
            user_embedding = torch.stack([torch.tensor(0.0) for _ in range(768)])
        else:
            user_embedding = torch.mean(torch.stack(looked_up_concept_embeddings),dim=0)
        user_known_concept_embeddings[uid] = user_embedding

    torch.save(user_known_concept_embeddings, "known_user_concept_embeddings.pt")

    # Create user embeddings tuple list (user id, user concept embedding)
    user_unknown_concept_embeddings = {user: [] for user in all_users}
    for uid in user_unknown_concept_dict.keys():
        looked_up_concept_embeddings = []
        for concept_id in user_known_concept_dict[uid]:
            concept_ids = concept_id.split("_")
            for x in concept_ids:
                looked_up_concept_embeddings.append(concept_embeddings[int(x)])
        if looked_up_concept_embeddings == []:
            user_embedding = torch.stack([torch.tensor(0.0) for _ in range(768)])
        else:
            user_embedding = torch.mean(torch.stack(looked_up_concept_embeddings),dim=0)
        user_unknown_concept_embeddings[uid] = user_embedding
    torch.save(user_unknown_concept_embeddings, "unknown_user_concept_embeddings.pt")
else:
    user_unknown_concept_embeddings = torch.load("unkown_user_concept_embeddings.pt")
    user_known_concept_embeddings = torch.load("known_user_concept_embeddings.pt")

print("All prepared files should exist now, exiting program")
