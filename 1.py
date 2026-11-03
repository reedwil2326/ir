# Text Pre-processing: lowercase -> tokenize -> stop-word removal -> stemming

import re
from nltk.stem import PorterStemmer                              # works offline, no download needed
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS   # built-in stop-word list

# Step 1: Read the input text
with open("data/docs/doc3.txt") as f:
    text = f.read()
print("Original :", text)

# Step 2: Lowercase, remove punctuation and tokenize
tokens = re.findall(r"[a-z]+", text.lower())
print("Tokens   :", tokens)

# Step 3: Stop-word removal
filtered = [w for w in tokens if w not in ENGLISH_STOP_WORDS]
print("No stop  :", filtered)

# Step 4: Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
print("Stemmed  :", stemmed)