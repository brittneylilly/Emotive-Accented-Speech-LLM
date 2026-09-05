'''
Script to get the count of unique characters and the count of unique words
in the .txt file  of transcripts that I will train the tokenizer on. 
Knowing the number of unique characters tells me the the minimum size 
that the vocubulary should be.

Run this script before training the tokenizer

You must know and set the predetermined vocubulary size in the tokenizer
training script. vocab size is set in my file train_tokenizer.py

#To run this script:
Open git bash
start the viritual environment
cd to main repo folder
run the script by typing python followed by the scripts path name:
python src/get_character_and_word_count.py

'''

#insert path of .txt file you want the unique character and word counts of
with open("data/transcripts_for_BPE_tokenization_training.txt", "r", encoding="utf-8") as f:
    text = f.read()

#punctation, whitespaces, and new lines are considered characters 
# and are thus included in the character count 
unique_characters = set(text)
print("Number of unique characters in input file:", len(unique_characters))

#words are split along whitespace or a new line. so a new word is evaluated each 
#time the algorithm comes to a whitespace or new line
words = text.split()
unique_words = set(words)
print("Number of unique words in input file:", len(unique_words))


'''
Output for transcripts_for_BPE_tokenization_training.txt file:
Number of unique characters in input file: 83
Number of unique words in input file: 10026
'''