# Practical 2: Document Retrieval using an Inverted Index (AND search)

import re
import glob
import os
from collections import defaultdict

# Step 1: Read all .txt files from data/docs/ -> {filename: text}
docs = {}
for path in sorted(glob.glob("data/docs/*.txt")):
    name = os.path.basename(path)[:-4]          # remove ".txt"
    with open(path) as f:
        docs[name] = f.read()

# Step 2: Build the inverted index -> word : set of documents containing it
index = defaultdict(set)
for name, text in docs.items():
    for word in re.findall(r"[a-z]+", text.lower()):
        index[word].add(name)

print("Index sample:", {w: sorted(index[w]) for w in ["documents", "web", "links"]})

# Step 3: AND search -> documents containing ALL the query words
def search(query):
    words = re.findall(r"[a-z]+", query.lower())
    sets = [index.get(w, set()) for w in words]
    return sorted(set.intersection(*sets)) if sets else []

# Step 4: Test queries
for q in ["documents", "web pages", "spam filters"]:
    print(f"Query '{q}' ->", search(q))
