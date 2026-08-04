# Emotive Accented Speech LLM
The Emotive Accented Speech LLM is a large language model built to generate text responses to prompts in the style of native Mandarin-speaking, English language learners, matching a selected emotional state: Neutral, Negative, or Pauses (which signals anxiety). The LLM is trained on human-verified transcripts of native Mandarin speakers, capturing the exact grammar and wording used when speaking English.

I built this LLM from scratch using a custom transformer architecture, trained entirely on the `EDEN ASR Dataset` that I manually verified, annotated, and labeled, rather than starting with an existing pre-trained model and fine-tuning it.

## Use Case
This LLM was created for researchers working in the areas of empathetic dialogue systems, accented speech research, spoken language processing, and machine learning. The original dataset used for this LLM only captured responses to a fixed set of prompts at one point in time, which limits what researchers can study. This LLM extends that dataset by allowing researchers to generate new, emotion-conditioned responses to any prompt they choose, including prompts about recent events or topics that were never a part of the original recordings. For example, a researcher could ask how a native Mandarin speaker learning English might respond to a prompt about a current event, and the model will generate a text-based response in the same authentic grammar and wording patterns found in the original transcripts of the dataset. This gives researchers a flexible tool to explore how native Mandarin speakers might express themselves in English across a much wider range of topics and situations than the original dataset alone could provide.

## Training Dataset

EDEN ASR Dataset 
Source: https://huggingface.co/datasets/sylviali/EDEN_ASR_Data 

EDEN is a robust open-domain chatbot for spoken conversation practice that provides empathetic feedback.

The dataset contains audio clips of native Mandarin speakers. The speakers conversed with the chatbot hosted on an English practice platform. 3081 audio clips from 613 conversations and 163 users remained after filtering. In the filtering process, I removed audio clips containing only Mandarin, duplicates, and a subset of self-introductions from the users. Each audio clip ranges from one second to two minutes. Demographic information was not collected to protect user identities. The speech was 
directly transcribed with Whisper Medium.  However, since the audio clips are accented speech, these transcripts have instances of ASR error. Thus, I manually verified and corrected the transcripts to match the exact grammar and words spoken by the user, overwriting the grammatically corrected version. This results in a high-quality, human-verified dataset.
