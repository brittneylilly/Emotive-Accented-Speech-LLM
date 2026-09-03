'''
Script to transfer the transcripts, along with their emotion label and Unit label
from the excel file into a JSON (via pandas/python) so that data can be used/readable
by the LLM
'''
#run from repo's main folder in Git Bash by typing: python src/extract_data.py
#you must ensure you're in the virtual environment first

import pandas as pd

#This script will only read the specific relevant columns from the excel sheet
df = pd.read_excel("data/Final_Transcripts.xlsx", usecols=["Unit", "emotion label", "corrected whisper_transcript"])

#use pandas str.replace() to replace all occurences of the ellipses "..." 
#pattern/regex with a blank space in the corrected whisper_transcript column
df["corrected whisper_transcript"] = df["corrected whisper_transcript"].str.replace("...", " ", regex=False)

#now use pandas str.replace() to remove any instances of more than one single space 
#within the string that could have occured as a result of replacing the ellipses with an empty space or
# double spaces that we accidnetally added due to human error. 
df["corrected whisper_transcript"] = df["corrected whisper_transcript"].str.replace(r"\s+", " ", regex=True)

#now use pandas str.strip() method to remove all trailing and leading spaces from 
#the transcript strings
df["corrected whisper_transcript"] = df["corrected whisper_transcript"].str.strip()


#convert the pandas dataframe (the data tables/columns) to JSON format,
# which is the format that will be used for training the LLM
data_as_json_string = df.to_json(orient="records")

#write the json_string to a file
with open("data/cleaned_transcripts.json", "w") as f:
    f.write(data_as_json_string)

