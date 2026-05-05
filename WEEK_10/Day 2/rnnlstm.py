import numpy as np
import pickle
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Training data with word associations
text = """ice cream is delicious apple juice is refreshing apple pie is sweet apple sauce is yummy
banana split is tasty chocolate cake is wonderful orange juice is healthy strawberry jam is good
deep learning is powerful deep learning is fun machine learning is great neural networks are powerful
peanut butter and jelly sandwich is classic coffee and milk morning routine is important"""

def train_model():
    """Train the LSTM model on the text"""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([text])
    
    vocab_size = len(tokenizer.word_index) + 1
    
    sequences = []
    for i in range(1, len(text.split())):
        seq = text.split()[:i+1]
        encoded = tokenizer.texts_to_sequences([" ".join(seq)])[0]
        sequences.append(encoded)
    
    max_len = max(len(seq) for seq in sequences)
    sequences = pad_sequences(sequences, maxlen=max_len)
    
    X = sequences[:, :-1]
    y = sequences[:, -1]
    
    model = Sequential([
        Embedding(vocab_size, 10, input_length=max_len-1),
        LSTM(100),
        Dense(vocab_size, activation='softmax')
    ])
    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    model.fit(X, y, epochs=100, verbose=1)
    
    model.save("lstm_model.h5")
    
    with open('tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    
    with open('max_len.pkl', 'wb') as f:
        pickle.dump(max_len, f)
    
    print("\n✓ Model trained and saved!")
    return model, tokenizer, max_len

def predict_next_word(input_text):
    """Predict the next word based on input text"""
    # Load model and tokenizer
    model = load_model("lstm_model.h5")
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    
    with open('max_len.pkl', 'rb') as f:
        max_len = pickle.load(f)
    
    # Process input
    input_text = input_text.strip().lower()
    encoded = tokenizer.texts_to_sequences([input_text])[0]
    padded = pad_sequences([encoded], maxlen=max_len-1)
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    predicted_word_index = np.argmax(prediction)
    
    # Convert index back to word
    word_index = {v: k for k, v in tokenizer.word_index.items()}
    predicted_word = word_index.get(predicted_word_index, "unknown")
    
    return predicted_word

# Train the model (comment out after first run)
if not os.path.exists("lstm_model.h5"):
    print("Training model...")
    train_model()
else:
    print("Model already exists. Skipping training...")

# Interactive prediction
print("\n" + "="*50)
print("NEXT WORD PREDICTOR")
print("="*50)
print("\nTry typing words like: ice, apple, coffee, banana, chocolate")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter text: ").strip()
    
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break
    
    if not user_input:
        print("Please enter some text!")
        continue
    
    predicted = predict_next_word(user_input)
    print(f"Input: '{user_input}' → Predicted next word: '{predicted}'\n")