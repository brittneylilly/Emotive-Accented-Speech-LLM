'''
Script to extract just the transcript texts from the json file of the cleaned transcripts
inorder to create a .txt of the transcript texts to be used in the tokenization training.
'''
#To Run Script
#use Git Bash and go to main repo directory
#activate virtual environment
# from main directory type: python src/extract_transcripttext_from_json.py

#this script will use pandas to read the json file and convert it back to a pandas dataframe
import pandas as pd

#convert the json file data back to a dataframe which will create a 
#separate column for the transcript text
df = pd.read_json("data/cleaned_transcripts.json", orient="records")

#keep/get only the column with the transcript text
df = df["corrected whisper_transcript"]

#write each row of the df column to the .txt file
with open("data/input_file_transcripts_for_training_tokenizer.txt", "w", encoding="utf-8") as f:
    for row in df:
        f.write(row + "\n")