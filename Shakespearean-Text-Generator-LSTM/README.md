# Shakespearean Text Generator (LSTM)

[![ Shakespearean Text Generator Demo](https://img.shields.io/badge/demo-animal--classifier-blue)](https://shakespearean-text-generator.streamlit.app/)

##  Project Overview
This project uses **LSTMs (Long Short-Term Memory networks)** to generate creative text. By training on Shakespeare's corpus, the model learns character-level patterns, allowing it to predict the most likely "next character" in a sequence.

##  Tech Stack
* **Language:** Python
* **Framework:** TensorFlow / Keras
* **Method:** Character-level RNN

##  Approach
1. **Data Pipeline:** Cleaned raw text and built a `char-to-int` mapping.
2. **Architecture:** Implemented an LSTM-based architecture designed to "remember" long-range dependencies, overcoming the vanishing gradient problem found in standard RNNs.
3. **Inference Loop:** Built a generation function that takes a seed phrase and iteratively predicts subsequent characters.

##  Results
By using LSTMs, the model maintains stylistic consistency over longer sequences, demonstrating a strong grasp of how recurrent structures handle sequential time-series data.