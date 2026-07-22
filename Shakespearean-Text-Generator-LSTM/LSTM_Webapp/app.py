import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os


st.set_page_config(
    page_title="Shakespearean Text Generator",
    page_icon="🖋️",
    layout="centered",
)

st.markdown("""
    <style>
    .stApp { background-color: #fbf7f0; }
    .generated-box {
        background-color: #fffaf0;
        border: 1px solid #b8860b;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        font-family: Georgia, serif;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .seed-highlight { color: #6b6b6b; }
    .generated-highlight { color: #b8860b; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "Shakespearean-Text-Generator-LSTM.keras")
tokenizer_path = os.path.join(base_dir, "tokenizer.pkl")


@st.cache_resource
def get_model():
    return tf.keras.models.load_model(model_path)

@st.cache_resource
def get_tokenizer():
    with open(tokenizer_path, 'rb') as f:
        return pickle.load(f)

model = get_model()
tokenizer = get_tokenizer()


with st.sidebar:
    st.header("🖋️ About")
    st.write(
        "An LSTM network trained on Shakespeare's writing, predicting one word "
        "at a time to continue a given starting phrase."
    )
    st.write("**Tech stack:** TensorFlow/Keras, Streamlit")
    st.write("**Try these starting phrases:**")
    st.code("this is the indictment of the")
    st.code("to be or not to")
    st.code("shall i compare thee to")


st.title("🖋️ Shakespearean Text Generator")
st.write("Enter a starting phrase, and the model will continue it in Shakespeare's style, one word at a time.")

user_input = st.text_input("Enter a starting phrase:", "this is the indictment of the")
next_words = st.slider("How many words to generate?", 1, 10, 5)

if st.button("Generate"):
    if user_input.strip() == "":
        st.warning("Please enter a starting phrase first!")
    else:
        seed_text = user_input
        max_sequence_len = 12
        generated_words = []

        progress = st.progress(0, text="Writing like Shakespeare...")

        for i in range(next_words):
            token_list = tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

            predicted_probs = model.predict(token_list, verbose=0)
            predicted_id = np.argmax(predicted_probs, axis=-1)[0]

            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted_id:
                    output_word = word
                    break

            seed_text += " " + output_word
            generated_words.append(output_word)
            progress.progress((i + 1) / next_words, text=f"Writing like Shakespeare... ({i + 1}/{next_words})")

        progress.empty()

        st.markdown(f"""
            <div class="generated-box">
                <span class="seed-highlight">{user_input}</span>
                <span class="generated-highlight"> {' '.join(generated_words)}</span>
            </div>
        """, unsafe_allow_html=True)

        st.caption("Gold text = newly generated words")