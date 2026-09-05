'''
Script to train the tokenizer (this tokenizer uses the BPE algorithm)
'''
#To run this script:
#Open git bash -> cd into projects root folder -> activiate virtual environemt
#then on commmand line type: python src/train_tokenizer.py

#My specific outputs from running this script are at the end

'''
This tokenizer training script takes in the .txt file of my transcript text and
outputs a json file that contains a dictionary with 2 parts: 

Part 1: a subdictionary called "vocab" that is the vocabulary dictionary, 
comprised of key:value pairs of vocabulary string:token ID
The keys are strings that are the special tokens, every individual character in 
the input text, and subwords and whole words that were merged (note a vocab
word will never be a word or subword that wasn't found in the input .txt file). 
The values are the token ID for every vocabulary string

Part 2: a sublist called "merges" that contains lists of strings of the two characters
and/or subwords that merged to form a vocabulary key. Each sublist only contains two 
strings bc how BPE worksd under the hood is that it only ever combines two things at a time.
Each sublist in this merge list shows the step by step history of how individuals characters 
were brought together to form a vocabulary key.
Each sublist in the merge list aligns to a specific multi-character vocabulary key. 
'''

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

#instantiate an new tokenizer object from the Tokenizer class
#The parameters configures the tokenizer behavior for encoding portion of tokenization (NOT the training).
#The special token parameter passed in here "[UNK]" has nothing to do with training the tokenizer.
# The reason I pass "[UNK]" now is that I'm setting up the BPE model to know how to handle the rare
# edgecase that if during encoding, there is a char, subword or word in my input .txt file that was
# not captured in my vocabulary output, I want the BPE tokenizer model to put "[UNK]" in place
# of that unrecognized word in the ouput of the encoding phase. 
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))



#Next, instantiate a pre_tokenizer to split the input text from the .txt file along white spaces and 
#various kinds punctation marks before the BpeTrainer starts doing the merges and creating the vocabulary dict.
#Using this pretokenizer to split along white spaces is necessary to define what the boundaries are 
#that define a word. Since we're uisng the BPE algorithm for subword tokenization that parses out 
#characters and builds them up to words, we need this training process to know the start and end point 
#of a new word (as opposed to it thinking an entire phrase or sentence could be a vocabulary word). 
#Remember the the vocabulary dictionary key will not contain any key larger than 1 word, and this is
#how we're designing it. 

#When the pretokenizer hits a new whitespace or various punctation marks like dashes etc, it defines
# a new marker and puts everythiing from that marker to the next marker into a tuple,
#along with the string's start index and end index +1, relative to the entire file. 
#The pretokenizer produces a list of tuples.
#For example, say the input .txt file has: "Hello! I'm fine."
# The pre_tokenizer does this:
# [("Hello",(0,5)), ("!", (5,6)), ("I", (7,8)), ("'", (8,9)), ("m", (9,10)), ("fine", (11,15)), (".", (15,16))]
# 
#Notice that the pretokenizer includes all punctuation as valid for tuple entries. This is confirmed
# by the fact that punctuation marks are vocabulary keys and have token IDs in the vocabulary dictionary.
# So the Whitespace() pretokenizer treats punctuation marks as its own "word". The only thing the 
# Whitespace() tokenizer actually will ignore from putting inside a tuple is an actual whitespace.
# and this is confirmed by the fact that a whitespace will never be a vocabulary key and have a token ID.
'''
#the Whitespace() pretokenizer splits input .txt file along whitespace and all characters that 
#are not letters, digits, or an underscore. 
'''
tokenizer.pre_tokenizer = Whitespace()



#Next, instantiate a new BpeTrainer object to train the tokenizer on my dataset, and 
#pass in parameters to set the vocabulary size;
#set the mininum frequency(aka the minimum number of times a merged string must
#occur in order for it to be made into a vocabulary word);
#and pass in the special tokens that I want the made into vocabs strings and assigned
#token IDs. 
# In this training phase, each special token will be added to the vocabulary dictionary
# and given a token ID. This way when the special tokens will have their token IDs for 
# use in the encoding phase (the encoding step is when the keys from the vocab dictionary
# are strung together as sequences or phrases and we represent these sequences/phrases
# as an array of token IDs. For example if the vocab key "the" has token ID 32 and the 
# vocab key "shirt" has token ID 59. Encoding would put these two vocab words togehter
# and represent it as the array [32, 59]). 

#The order of the special tokens in the BpeTrainer parameter determines the special tokens's 
#token ID number in the vocabulary dictionary. So since "[UNK]" is the first special token 
# listed, then it's token ID will be 0. 

# What are special tokens: 
# Special tokens strings that I want to show up in the encoding phase output that will later
# help when training the actual LLM with pattern recognization and response formation. 

# since the vocabulary keys are strings, the special tokens must be strings in order to be 
# assiged a token ID like all other vocab keys.
# It is important to know that the special tokens themselves have no actual meaning translative
# meaning to anything in this entire LLM buidling process.Like nothing
# mis going to interpret UNK, for example, as an instruction to go do something.

# "[UNK]" stands for unknown token. We are telling the tokenizer, during encoding, in the 
# rare event you come across a string in the .txt transcript text file that is not a key in the 
# vocabulary dictionary, then in the encoding array use UNK'S token ID instead of that unrecognized string.

# "[PAD]" stands for padding. PAD's token ID will be used to make the encoded arrays of token sequences
# the same size. (Ensuring that the arrays are the same size is necessary for a later part in the 
# LLM buidling process when the we feed the token sequence array to the transformer. A bunch of these
# arrays are scooped up together and given to the transfomer in batches,  but the transformer requires 
# the contents that its processing to be in a matrix, and the matrix must be perfectly rectangle/square 
# in order for the matrix math to work. so this means each array the batch that's passed to the 
# transformer has to be the same length. Thus "[PAD]"'s token ID is added at the end of array token sequences
# as needed so that shorter arrays can be as long as the longest array in that batch.

# "[EOS]" stands for end of sentence. Note that EOS doesn't replace
# periods or other chars that end a sentence. Those are still there. 
# We use "[EOS]" to indicate when one response in the transcript ends (and thus another one begins). 
# eg. "I" "always" "go" "to" "zoo" "." "[EOS]" "Nobody" "."

# I have to write a script to manually insert "[EOS]" into my .txt file before i give it to the 
# trained tokenizer in the encoding phase. 

# Why is the EOS specal token necessary:
# Later in the process of building my LLM when it is learning what a completed response is,
# by using statistical pattern recognition it'll learn this by pick up that every time "[EOS]" is there, 
# the sequence of strings before "[EOS]" is completely unrelated th the sequence of strings after it. 
# So having this will help my model learn how a complete response starts and ends. It's important to note
# that just like the other special tokens, the string phrase "[EOS]" has no actual meaning.
'''
The BpeTrainer will stop running once it reaches the vocab_size OR once it
runs out of vocab words to form from the merge strings based on the 
min_frequency threshold I set. 
I'm setting my vocabulary to 30,000 based on Hugging Face quicktour doc and 
Sebastian Raschka article, stating 30k and 32k as the min vocab size of modern LLMs. 
Although my vocab size will certainly be significantly smaller, I will guard my tokenizer 
training from turning one-off or meaningless words from my input data into vocabulary words,
by setting the min_frequency to 2. A min_frequencey of 2 means that during 
training, if a merged string only appears 1 time, it will not be turned into 
a vocabulary key/word. The merged string must appear at least 2 times to be considered
a vocabulary word and given a token ID.
'''
trainer = BpeTrainer(vocab_size=30000, min_frequency=2,special_tokens=["[UNK]", "[PAD]", "[EOS]"]) 



#Identify the file path of the .txt file that you want to use for training the tokenizer
#and save the file path as a variable 
file = ["data/input_file_transcripts_for_training_tokenizer.txt"]



#Now we're ready to run the training of the tokenizer
#This line runs/starts the tokenizer training.
#Run the tokenizer.train() method to Train the tokenizer on the file.
#The two parameters we pass in are the variable name of the .txt file path I want to train the 
# tokenizer on (variable named on previous line).
#And the 2nd parameter is the actual BpeTrainer object that I created/instantiated in the earlier 
# line with BpeTrainer.
#After this line is executed, we will have a trained tokenizer
tokenizer.train(file, trainer)



#Call the get_vocab_size method on my tokenizer object so I can get the actual
#size if my vocabulary after running the trainer. I know my actual vocab size will 
#be much smaller than the threshold I set for it when i configured the BpeTrainer. So I 
#need to know that actual vocab_size bc I will use it for buidling the embedding layer later on.
'''
The vocab size is the number of keys in the vocabulary dictionary.
When I run this script, the vocab size will print in the terminal on the last line. 
#Regarding the with_added_tokens parameter set to True, this just means that the 
#vocab size will include the number of special tokens I defined earlier(UNK, PAD, EOS). I only
#used 3 specal tokens so they only add +3 in the vocab size count.
'''
print(tokenizer.get_vocab_size(with_added_tokens=True))
#I ran the script and my vocab size is 6,131


#Indicate the name of the file and file path where you want the 
#output file of running the training saved. YOu do not need to precreate
#the file. if it doesn't already exist, it will be created when this line runs.
'''
The output of tokenizer training is a dictionary that contains two parts:
a subdictinary called "vocab" that is the vocabulary dictionary, 
and a sublist called "merges" that contains lists of strings of the character 
merges that formed a vocab key. 
'''
tokenizer.save("data/output_vocabulary_file_from_trained_tokenizer.json") 



'''
My output from running the script:

Vocabulary size (aka number of keys in the vocab dict): 6,131

Count of words/punctionations formed as tuples by the pretokenizer: 6,423

Counted pairs: 6,423  #this is the count of unique words/subwords being analyzed to determine
if they can formed with something else a merged pair

Number of merge pairs formed after meeting min_frequency: 6,047  #this is equal to the number 
of sublists in the merges list

'''