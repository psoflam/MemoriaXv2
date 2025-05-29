from typing import List
import faiss
import numpy as np

class MemoryIndex:
    def __init__(self, embedding_dim: int = 384):  # Optional default value to fix tests
        print("Init called with dim:", embedding_dim)  # Optional debug
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.id_map = {}

    def add_memory(self, id: str, vector: List[float]):
        self.index.add(np.array([vector], dtype=np.float32))
        self.id_map[self.index.ntotal - 1] = id

    def query_similar(self, vector: List[float], top_k: int) -> List[str]:
        D, I = self.index.search(np.array([vector], dtype=np.float32), top_k)
        return [self.id_map[i] for i in I[0] if i != -1]

    def rebuild_index(self):
        # Placeholder for batch update logic
        pass 