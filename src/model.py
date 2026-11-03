'''
The model code for my small GPT-style language model.
The app (and any test script) imports from this file.

MODEL_CONFIG must match the settings the saved model files were trained with,
otherwise the saved numbers won't fit into the model.
'''

import re
import torch

# The same settings I used in the Colab notebook
MODEL_CONFIG = {
    "vocab_size": 6134,
    "context_length": 64,
    "emb_dim": 128,
    "n_heads": 4,
    "n_layers": 4,
    "drop_rate": 0.2,
}


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, emb_dim, n_heads, context_length, drop_rate):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.query = torch.nn.Linear(emb_dim, emb_dim)
        self.key = torch.nn.Linear(emb_dim, emb_dim)
        self.value = torch.nn.Linear(emb_dim, emb_dim)
        self.out = torch.nn.Linear(emb_dim, emb_dim)
        self.dropout = torch.nn.Dropout(drop_rate)
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        self.register_buffer("mask", mask)

    def forward(self, x):
        batch, length, emb_dim = x.shape
        q = self.query(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(2, 3) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.mask[:length, :length], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        result = (weights @ v).transpose(1, 2).reshape(batch, length, emb_dim)
        return self.out(result)


class FeedForward(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 4 * emb_dim),
            torch.nn.GELU(),
            torch.nn.Linear(4 * emb_dim, emb_dim),
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(config["emb_dim"])
        self.attention = CausalSelfAttention(config["emb_dim"], config["n_heads"], config["context_length"], config["drop_rate"])
        self.norm2 = torch.nn.LayerNorm(config["emb_dim"])
        self.feed_forward = FeedForward(config["emb_dim"])
        self.dropout = torch.nn.Dropout(config["drop_rate"])

    def forward(self, x):
        x = x + self.dropout(self.attention(self.norm1(x)))
        x = x + self.dropout(self.feed_forward(self.norm2(x)))
        return x


class SmallGPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.position_embedding = torch.nn.Embedding(config["context_length"], config["emb_dim"])
        self.dropout = torch.nn.Dropout(config["drop_rate"])
        self.blocks = torch.nn.Sequential(*[TransformerBlock(config) for _ in range(config["n_layers"])])
        self.final_norm = torch.nn.LayerNorm(config["emb_dim"])
        self.output_layer = torch.nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, ids):
        batch, length = ids.shape
        positions = torch.arange(length, device=ids.device)
        x = self.token_embedding(ids) + self.position_embedding(positions)
        x = self.dropout(x)
        x = self.blocks(x)
        x = self.final_norm(x)
        return self.output_layer(x)


def load_model(checkpoint_path, device="cpu"):
    # Build an empty model with the same settings, then fill it with the saved numbers
    model = SmallGPT(MODEL_CONFIG)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate(model, tokenizer, start_text, max_new_tokens=40, temperature=0.8, top_k=20):
    device = next(model.parameters()).device
    eos_id = tokenizer.token_to_id("[EOS]")

    ids = tokenizer.encode(start_text).ids
    if len(ids) == 0:
        ids = [eos_id]
    ids = torch.tensor([ids], device=device)

    for _ in range(max_new_tokens):
        context = ids[:, -MODEL_CONFIG["context_length"]:]
        with torch.no_grad():
            logits = model(context)[:, -1, :]
        logits = logits / temperature
        top_values, _ = torch.topk(logits, top_k)
        logits[logits < top_values[:, -1:]] = float("-inf")
        probabilities = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probabilities, num_samples=1)
        if next_id.item() == eos_id:
            break
        ids = torch.cat([ids, next_id], dim=1)

    return tokenizer.decode(ids[0].tolist())


def tidy_text(text):
    # Remove the space the tokenizer puts before punctuation: "me ." becomes "me."
    text = re.sub(r"\s+([.,?!;:])", r"\1", text)
    # Join apostrophes to the letters around them: "don ' t" becomes "don't"
    text = re.sub(r"\s*'\s*", "'", text)
    return text