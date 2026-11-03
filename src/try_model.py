from tokenizers import Tokenizer
from model import load_model, generate, tidy_text

tokenizer = Tokenizer.from_file("models/tokenizer.json")
neutral_model = load_model("models/neutral_model.pt")
negative_model = load_model("models/negative_model.pt")

for name, model in [("Neutral", neutral_model), ("Negative", negative_model)]:
    print(name)
    for start in ["I think", "I don't", "Yes"]:
        text = generate(model, tokenizer, start)
        print("  ", start, "->", tidy_text(text))
    print()