import networkx as nx
import numpy as np
from gensim.models import Word2Vec
import random
import pickle
from pathlib import Path

def load_graph():
    """
    Loads the ICD-10 Graph
    """
    # Go up 4 levels: embeddings -> preprocessing -> utils -> src -> ROOT
    root_dir = Path(__file__).resolve().parents[4]
    file_path = root_dir / "data" / "icd10_graph.pkl"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Graph not found at {file_path}")    
    with open(file_path, "rb") as f:
        return pickle.load(f)

def get_alias_edge(G, t, v, p, q):
    """
    Calculates unnormalized transition probabilities for the bias (p/q).
    Logic: t -> v -> x (next candidate)
    """
    unnormalized_probs = [] #Initializes an empty list to store the raw, unweighted probabilities for each candidate neighbor.
    candidates = list(G.neighbors(v))
    
    for x in candidates:
        weight = G[v][x].get('weight', 1.0)
        
        if x == t:  # Return to previous node (t)
            unnormalized_probs.append(weight * (1.0 / p))
        elif G.has_edge(x, t):  # Neighbor of previous
            unnormalized_probs.append(weight * 1.0)
        else:  # Not connected to previous (Explore outward)
            unnormalized_probs.append(weight * (1.0 / q))
            
    return candidates, unnormalized_probs

def simulate_walks(G, num_walks=10, walk_length=20, p=1.0, q=1.0):
    walks = []
    nodes = list(G.nodes())    
    # Use fast path if parameters are neutral
    is_unbiased = (p == 1.0 and q == 1.0)
    
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            
            while len(walk) < walk_length:
                cur = walk[-1]
                neighbors = list(G.neighbors(cur))
                if not neighbors: 
                    break
                if is_unbiased:
                    # Case A: Fast Random Walk (O(1))
                    next_node = random.choice(neighbors)
                else:
                    # Case B: Biased Node2Vec Walk
                    if len(walk) == 1:
                        # First step has no history, treat as unbiased
                        next_node = random.choice(neighbors)
                    else:
                        prev = walk[-2]
                        # Get weighted probabilities based on p and q
                        candidates, weights = get_alias_edge(G, prev, cur, p, q)
                        # Weighted random choice (k=1 returns a list, so we take [0])
                        next_node = random.choices(candidates, weights=weights, k=1)[0]            
                walk.append(next_node)
            walks.append(walk)           
    return walks

def get_node2vec_embeddings(vector_size=100, num_walks=10, walk_length=20, p=1.0, q=1.0, window=5, workers=4):
    """
    Generate embeddings from the graph.
    """
    # 1. Load Cached Graph
    G = load_graph()
    
    # 2. Simulate Walks (Logic handles p/q internally)
    walks = simulate_walks(G, num_walks, walk_length, p, q)
    
    # 3. Train Model (Scalable window/workers)
    model = Word2Vec(
        sentences=walks, 
        vector_size=vector_size, 
        window=window,
        min_count=1, 
        sg=1, 
        workers=workers,
        seed=42
    ) 
    # 4. Extract KeyedVectors
    embedding_dict = {}
    for word in model.wv.index_to_key:
        embedding_dict[word] = model.wv[word]
    return embedding_dict

def train_node2vec(sentences, vector_size, num_walks=10, walk_length=20, p=1.0, q=1.0, window=5, workers=4):

    return get_node2vec_embeddings(
        vector_size=vector_size, 
        num_walks=num_walks, 
        walk_length=walk_length,
        p=p,
        q=q,
        window=window,
        workers=workers
    )