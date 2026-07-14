import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

# -----------------------------------
# 1. English Sentences List
# -----------------------------------

sentences = [
    "I love deep learning",
        "RNN is powerful",
            "Neural networks are amazing",
                "AI is the future",
                    "Machine learning is fun",
                        "I enjoy coding",
                            "Python is easy",
                                "Deep learning uses data",
                                    "RNN remembers previous information",
                                        "AI can solve problems"
                                        ]
# Labels (Example)
# 1 = Positive / Important
# 0 = Negative / Less Important

labels = np.array([1,1,1,1,1,1,1,1,1,1]) # Converted to numpy array

# -----------------------------------
# 2. Convert Words → Numbers
# -----------------------------------
print(sentences)
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

print(sentences)

sequences = tokenizer.texts_to_sequences(sentences)


print(tokenizer.word_index)
print(sequences)

# -----------------------------------
# 3. Padding
# -----------------------------------

max_length = 6

padded_sequences = pad_sequences(
    sequences,
        maxlen=max_length,
            padding='post'
            )
print("\nPadded Sequences:")
print(padded_sequences)

# -----------------------------------
# 4. Build RNN Model
# -----------------------------------

vocab_size = len(tokenizer.word_index) + 1

model = Sequential([

    # Word Embedding Layer
        Embedding(input_dim=vocab_size,
                      output_dim=16,
                                    input_length=max_length),
    # RNN Layer
        SimpleRNN(32),
    # Output Layer
        Dense(1, activation='sigmoid')
        ])
# -----------------------------------
# 5. Compile Model
# -----------------------------------

model.compile(
    optimizer='adam',
        loss='binary_crossentropy',
            metrics=['accuracy']
            )
# -----------------------------------
# 6. Train Model
# -----------------------------------

model.fit(
    padded_sequences,
        labels,
            epochs=20
            )
# -----------------------------------
# 7. Test New Sentence
# -----------------------------------

test_sentence = [" AI is amazing "]

test_seq = tokenizer.texts_to_sequences(test_sentence)

test_pad = pad_sequences(
    test_seq,
        maxlen=max_length,
            padding='post'
            )
prediction = model.predict(test_pad)

print("\nPrediction:", prediction)
