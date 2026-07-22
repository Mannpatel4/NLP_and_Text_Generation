import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Sequential Chatbot",
    page_icon="💬",
    layout="centered",
)

st.markdown("""
    <style>
    .stApp { background-color: #f5f7fb; }
    .chat-bubble-user {
        background-color: #6366f1;
        color: white;
        padding: 0.7rem 1rem;
        border-radius: 14px 14px 2px 14px;
        margin: 0.4rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bubble-bot {
        background-color: #e5e7eb;
        color: #1a1a1a;
        padding: 0.7rem 1rem;
        border-radius: 14px 14px 14px 2px;
        margin: 0.4rem 0;
        max-width: 80%;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_model():
    return load_model("Chatbot.keras")

@st.cache_resource
def get_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def get_encoder():
    with open("encoder.pkl", "rb") as f:
        return pickle.load(f)

model = get_model()
tokenizer = get_tokenizer()
encoder = get_encoder()

max_length = 6 


with st.sidebar:
    st.header("💬 About")
    st.write(
        "A sequence-based chatbot built with an LSTM network trained to classify "
        "user questions into predefined response categories."
    )
    st.write("**Tech stack:** TensorFlow/Keras, Streamlit")
    st.caption("This is an intent-classification chatbot, not a generative LLM — it maps your question to the closest trained response.")


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.title("💬 Sequential Chatbot")
st.write("Ask something below — the model will classify your question and respond.")


for role, message in st.session_state.chat_history:
    bubble_class = "chat-bubble-user" if role == "user" else "chat-bubble-bot"
    st.markdown(f'<div class="{bubble_class}">{message}</div>', unsafe_allow_html=True)

question = st.text_input("Ask something", key="question_input")

col1, col2 = st.columns([1, 5])
with col1:
    send = st.button("Send")
with col2:
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

if send and question.strip():
    with st.spinner("Thinking..."):
        seq = tokenizer.texts_to_sequences([question])
        seq = pad_sequences(seq, maxlen=max_length, padding="post")

        prediction = model.predict(seq, verbose=0)
        index = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        answer = encoder.inverse_transform([index])[0]

    st.session_state.chat_history.append(("user", question))
    st.session_state.chat_history.append(("bot", answer))
    st.caption(f"Confidence: {confidence:.1f}%")
    st.rerun()
elif send:
    st.warning("Please type a question first!")