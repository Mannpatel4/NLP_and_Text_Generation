# Sequential Chatbot

[![Sequential Chatbot Demo](https://img.shields.io/badge/demo-Sequential--Chatbot-blue)](https://sequential-chatbot-lstm.streamlit.app/)

##  Project Overview
A chatbot designed to process user inputs sequentially and generate appropriate responses, demonstrating foundational mechanics of conversational AI.

##  Tech Stack
* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Processing:** Tokenization, Padding, and Embedding layers

##  Approach
1. **Preprocessing:** Converted natural language into numeric sequences using `Tokenizer` and `pad_sequences` to ensure uniform input shapes.
2. **Model Design:** Designed a sequential architecture to handle short-term conversational context.
3. **Response Generation:** Created a logic flow that maps input sequences to the highest-probability output sequences.

##  Results
Successfully built a model that maintains basic conversational flow, showcasing an understanding of sequence-to-sequence data handling and NLP fundamentals.