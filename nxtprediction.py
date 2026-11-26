import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

sentences = [
    "I love machine learning",
    "I love deep learning",
    "I love artificial intelligence",
    "machine learning is powerful",
    "deep learning is powerful",
    "artificial intelligence is powerful",
    "machine learning is useful",
    "deep learning is useful"
]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)

total_words = len(tokenizer.word_index) + 1

print("Word Index:")
print(tokenizer.word_index)


input_sequences = []

for sentence in sentences:

    sequence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(sequence)):
        input_sequences.append(sequence[:i + 1])

print("\nSequences:")
print(input_sequences)


max_sequence_length = max(len(seq) for seq in input_sequences)

input_sequences = pad_sequences(
    input_sequences,
    maxlen=max_sequence_length,
    padding="pre"
)


X = input_sequences[:, :-1]

y = input_sequences[:, -1]


y = tf.keras.utils.to_categorical(
    y,
    num_classes=total_words
)

print("\nX shape:", X.shape)
print("y shape:", y.shape)


model = Sequential([
    Embedding(
        input_dim=total_words,
        output_dim=64,
        input_length=max_sequence_length - 1
    ),

    LSTM(128),

    Dense(total_words, activation="softmax")
])


model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

model.summary()


model.fit(
    X,
    y,
    epochs=200,
    verbose=1
)


def predict_next_word(seed_text):

    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    if not token_list:
        return None

    sequence = pad_sequences(
        [token_list],
        maxlen=max_sequence_length - 1,
        padding="pre"
    )

    prediction = model.predict(sequence, verbose=0)

    predicted_word_index = np.argmax(prediction)

    return tokenizer.index_word.get(predicted_word_index, None)



if __name__ == "__main__":
    print("\n--- Next Word Prediction ---")
    while True:
        user_input = input("\nEnter input text (or type 'exit' to quit): ").strip()

        if user_input.lower() in ["exit", "quit"]:
            print("Exited.")
            break

        if not user_input:
            print("Please enter non-empty text.")
            continue

        predicted_word = predict_next_word(user_input)

        if predicted_word:
            print(f"Predicted Next Word: {predicted_word}")
            print(f"Result: {user_input} {predicted_word}")
        else:
            print("Could not predict next word (words not in vocabulary).")