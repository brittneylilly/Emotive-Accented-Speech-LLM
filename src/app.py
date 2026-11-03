import streamlit as st
from pathlib import Path
from tokenizers import Tokenizer
from model import load_model, generate, tidy_text

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@st.cache_resource
def load_everything():
    tokenizer = Tokenizer.from_file(str(MODELS_DIR / "tokenizer.json"))
    models = {
        "Neutral": load_model(str(MODELS_DIR / "neutral_model.pt")),
        "Negative": load_model(str(MODELS_DIR / "negative_model.pt")),
    }
    return tokenizer, models


tokenizer, models = load_everything()

st.title("Emotive, Accented Speech Language Model")
st.write(
    "Pick an emotion, type the start of a sentence, and the language model will complete the sentence. "
    "This language model was trained on transcripts of native Mandarin speakers practicing spoken English," 
    "so the generated sentences imitate their phrasing and grammar patterns, including repeated words and restarts."  
)


STARTERS = ["I think", "I don't", "Yes", "Because"]

if "start_text" not in st.session_state:
    st.session_state.start_text = ""

emotion = st.selectbox("Pick an Emotion", ["Neutral", "Negative"])

st.write("Pick a starter, or type your own below:")
columns = st.columns(len(STARTERS))
for column, starter in zip(columns, STARTERS):
    if column.button(starter):
        st.session_state.start_text = starter

start_text = st.text_input("Start of a sentence", key="start_text")


if st.button("Complete the sentence"):
    if 0 in tokenizer.encode(start_text).ids:
        st.warning("Some of your characters are not in the model's vocabulary and will be treated as unknown.")
    st.write(f"**{emotion}** completions:")
    for _ in range(3):
        text = generate(models[emotion], tokenizer, start_text)
        st.write("- " + tidy_text(text))