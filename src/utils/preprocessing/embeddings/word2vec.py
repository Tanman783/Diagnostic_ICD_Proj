from gensim.models import Word2Vec

def train_word2vec(sequences_dict, vector_size=100, window=5, min_count=1, workers=4):
    """
    Trains a Word2Vec model.
    """
    print(f"🔄 Training Word2Vec (Dim={vector_size}, Window={window})...")
    
    # sequences_dict.values() gives the list of lists needed for sentences
    sentences = list(sequences_dict.values())
    
    model = Word2Vec(
        sentences=sentences, 
        vector_size=vector_size, 
        window=window, 
        min_count=min_count, 
        sg=1, 
        workers=workers, 
        seed=42
    )
    
    print("Word2Vec Training Complete.")
    return model