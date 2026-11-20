# Install: pip install gensim

from gensim.models import Word2Vec

# Sample sentences
sentences = [
    ["deep", "learning", "is", "interesting"],
    ["machine", "learning", "is", "useful"],
    ["deep", "learning", "uses", "neural", "networks"]
]

# Train Word2Vec model
model = Word2Vec(sentences, vector_size=10,
                 window=2, min_count=1)

# Get word embedding
print("Embedding:")
print(model.wv["learning"])

# Find similar words
print("Similar words:")
print(model.wv.most_similar("learning"))