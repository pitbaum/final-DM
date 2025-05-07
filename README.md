https://www.kaggle.com/competitions/2025-data-mining-final-project/leaderboard

Run the data_preparation.py first, it will create several files that are loadable such that the embeddings etc dont need to be run every time before training a model

-- Using an embedding from a loaded 3rd party encoder
-- Encoding the concatinated insides of the questions
    -- Taking the mean of all question embeddings a user anwsered correctly to get user known question embeddings
    -- Taking the mean of all queston embeddings a user anwsered wrongly to get the user unkown question embeddings

-- Encoding all concpets with the same encoder
    -- Taking the mean of all concepts a user anwsered a question too correctly to get the user known concept embeddings
    -- Taking the mean of all concepts a user anwsered a question too wrongly to get the user unknown concept embeddings
    
Those can then later be fed into some neural network to evaluate if a user will anwser some question correctly or not

