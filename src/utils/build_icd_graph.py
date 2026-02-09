import sys
import pickle
import networkx as nx
import simple_icd_10 as icd
from pathlib import Path

# SETUP PATHS

current_file = Path(__file__).resolve()
# Path: src/utils/build_icd_graph.py -> src/utils -> src -> ROOT
project_root = current_file.parents[2]
data_dir = project_root / "data"
output_path = data_dir / "icd10_graph.pkl"

data_dir.mkdir(parents=True, exist_ok=True)

def format_code(code):
    """
    Standardizes codes to match MIMIC format.
    - Removes dots (e.g., 'E11.9' -> 'E119'), Strips whitespace
    """
    if not isinstance(code, str):
        return str(code)
    return code.replace(".", "").strip()

# BUILD GRAPH (The Ontology)

def build_icd10_graph():
    G = nx.Graph()
    
    # 1. Get every code known to the library
    all_codes = icd.get_all_codes() 
    print(f"Library Standard: Found {len(all_codes)} codes.")

    count = 0
    for code in all_codes:
        # Normalize the child
        child_node = format_code(code)
        G.add_node(child_node)
        
        # 2. Get the Parent
        parent = icd.get_parent(code)
        
        if parent:
            # Normalize the parent
            parent_node = format_code(parent)
            
            # 3. Create the Edge
            G.add_node(parent_node) # Ensure parent exists
            G.add_edge(child_node, parent_node)
            
        count += 1
        if count % 5000 == 0:
            print(f"   Processed {count} codes...")

    # Statistics
    print(f"\nGraph Construction Complete.")
    print(f"Total Nodes: {G.number_of_nodes()}")
    print(f"Total Edges: {G.number_of_edges()}")
    
    return G

# EXECUTION

if __name__ == "__main__":
    try:
        graph = build_icd10_graph()
        
        # Quality Check
        if graph.number_of_nodes() < 1000:
            print("\n[CRITICAL WARNING] The graph is suspiciously small.")
        else:
            with open(output_path, "wb") as f:
                pickle.dump(graph, f)
            print(f"\nSUCCESS: Graph saved to {output_path}")
            print("Ready for Node2Vec.")
            
    except Exception as e:
        print(f"FAILED: {e}")