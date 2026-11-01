'''
Script to turn my transcripts into numbers (token IDs) the model can train on.
Splits off 10% of the transcripts for measuring, adds [EOS] to the end of each
transcript, encodes everything, and saves two files.

To run: Git Bash -> cd to main repo folder -> activate venv ->
python src/encode_dataset.py
'''

import random
import numpy as np
from tokenizers import Tokenizer

# 1. Load the trained tokenizer from the file I saved in Step 8b
tokenizer = Tokenizer.from_file("data/output_vocabulary_file_from_trained_tokenizer.json")

# 2. Read the transcripts, one per line, and skip any blank lines
with open("data/input_file_transcripts_for_training_tokenizer.txt", "r", encoding="utf-8") as f:
    transcripts = [line.strip() for line in f if line.strip()]

# 3. Shuffle, then set aside 10% for measuring
random.seed(42)
random.shuffle(transcripts)
number_to_set_aside = int(len(transcripts) * 0.10)
measuring_transcripts = transcripts[:number_to_set_aside]
training_transcripts = transcripts[number_to_set_aside:]

# 4. Add [EOS] to the end of every transcript
measuring_transcripts = [t + " [EOS]" for t in measuring_transcripts]
training_transcripts = [t + " [EOS]" for t in training_transcripts]

# 5. Encode every transcript (turn text into lists of token IDs)
measuring_encoded = tokenizer.encode_batch(measuring_transcripts)
training_encoded = tokenizer.encode_batch(training_transcripts)

# 6. Checks
assert tokenizer.token_to_id("[EOS]") == 2
assert all(e.ids[-1] == 2 for e in training_encoded)
assert all(e.ids[-1] == 2 for e in measuring_encoded)
assert tokenizer.encode("[NEGATIVE] I think so.").ids[0] == 4

# 7. Join all the IDs into one long list per file
training_ids = [i for e in training_encoded for i in e.ids]
measuring_ids = [i for e in measuring_encoded for i in e.ids]

print("Training transcripts:", len(training_transcripts))
print("Measuring transcripts:", len(measuring_transcripts))
print("Training tokens:", len(training_ids))
print("Measuring tokens:", len(measuring_ids))
print("[UNK] tokens in training:", training_ids.count(0))

# 8. Save both lists to files
np.save("data/train_token_ids.npy", np.array(training_ids, dtype=np.uint16))
np.save("data/measuring_token_ids.npy", np.array(measuring_ids, dtype=np.uint16))