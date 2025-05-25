https://www.kaggle.com/competitions/2025-data-mining-final-project/leaderboard

Run the data_preparation.py first, it will create several files that are loadable such that the embeddings etc dont need to be run every time before training a model

-- Using an embedding from a loaded 3rd party encoder
-- Encoding the concatinated insides of the questions, stored in the file question embeddings

-- Encoding all concpets with the another or same embedding encoder: storing in the concept embeddings file
    
Those can then later be fed into some neural network to evaluate if a user will anwser some question correctly or not

-- Network consists of question_embedding asked, concept embedding of the question, the user_id
    -- A encoding layer that takes the user_id and learns a user encoding during the training step
    -- A concept encoding layer that takes the user_id and learns a user encoding during the trainig step
    -- Additional feed forward neural network layers with Dropout and binary output, that processes the user question and concept encodings in comparison to the question asked


- For easier training and testing pipeline, the whole workload has been put into the one_file_run.py

Report Link (WIP)
https://docs.google.com/document/d/191LyYo-p80vasWuxCMtK70bJHh5C_9CGdrVog26R7-0/edit?usp=sharing