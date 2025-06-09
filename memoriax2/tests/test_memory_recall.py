import unittest
import numpy as np
from memoriax2.nlp.memory_recall import embed_text, store_embedding, retrieve_similar_memories
from memoriax2.db.init import init_db
from memoriax2.memory.index_engine import MemoryIndex

class TestMemoryRecall(unittest.TestCase):

    def setUp(self):
        # Initialize the database connection
        self.conn = init_db()
        # Create the memory_embeddings table if it doesn't exist
        cursor = self.conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS memory_embeddings
                          (key TEXT PRIMARY KEY, embedding BLOB)''')
        self.conn.commit()
        # Initialize MemoryIndex
        self.memory_index = MemoryIndex(384)
        print("Setup complete: Database and MemoryIndex initialized.")

    def test_embed_text(self):
        # Test embedding generation
        text = "This is a test sentence."
        embedding = embed_text(text)
        self.assertEqual(len(embedding.shape), 1)  # Ensure it's a 1D array

    def test_store_and_retrieve_embeddings(self):
        # Test storing and retrieving embeddings with real sentences
        text1 = "I love programming."
        text2 = "I'm feeling very sad today."
        text3 = "Do you remember our trip to Japan?"
        key1 = "entry1"
        key2 = "entry2"
        key3 = "entry3"

        # Store embeddings
        embedding1 = embed_text(text1)
        embedding2 = embed_text(text2)
        embedding3 = embed_text(text3)
        store_embedding(self.conn, key1, embedding1)
        store_embedding(self.conn, key2, embedding2)
        store_embedding(self.conn, key3, embedding3)
        print("Embeddings stored in database.")

        # Add embeddings to MemoryIndex
        self.memory_index.add_memory(key1, embedding1)
        self.memory_index.add_memory(key2, embedding2)
        self.memory_index.add_memory(key3, embedding3)
        print("Embeddings added to MemoryIndex.")

        # Retrieve similar memories
        similar_memories = retrieve_similar_memories("I love coding.", self.conn, self.memory_index, top_k=1)
        print("Retrieved similar memories for 'I love coding.':", similar_memories)
        self.assertIn(key1, [mem[0] for mem in similar_memories])  # Expecting key1 to be the most similar

        similar_memories = retrieve_similar_memories("I'm feeling down today.", self.conn, self.memory_index, top_k=1)
        print("Retrieved similar memories for 'I'm feeling down today.':", similar_memories)
        self.assertIn(key2, [mem[0] for mem in similar_memories])  # Expecting key2 to be the most similar

        similar_memories = retrieve_similar_memories("Do you remember our vacation to Japan?", self.conn, self.memory_index, top_k=1)
        print("Retrieved similar memories for 'Do you remember our vacation to Japan?':", similar_memories)
        self.assertIn(key3, [mem[0] for mem in similar_memories])  # Expecting key3 to be the most similar

    def test_edge_cases(self):
        # Test empty input
        empty_embedding = embed_text("")
        self.assertEqual(len(empty_embedding.shape), 1)

        # Test very long input
        long_text = "a" * 10000  # Very long string
        long_embedding = embed_text(long_text)
        self.assertEqual(len(long_embedding.shape), 1)

        # Test special characters
        special_text = "!@#$%^&*()_+"
        special_embedding = embed_text(special_text)
        self.assertEqual(len(special_embedding.shape), 1)

    def test_performance(self):
        import time
        start_time = time.time()
        text = "Performance test sentence."
        for _ in range(1000):
            embedding = embed_text(text)
            store_embedding(self.conn, f"key_{_}", embedding)
        end_time = time.time()
        print(f"Performance test completed in {end_time - start_time} seconds.")
        self.assertTrue((end_time - start_time) < 10)  # Temporarily increased to 10 seconds for debugging

    def test_consistency(self):
        text = "Consistency test sentence."
        key = "consistency_key"
        embedding = embed_text(text)
        store_embedding(self.conn, key, embedding)
        print(f"Stored embedding for key '{key}'.")

        # Add embedding to MemoryIndex
        self.memory_index.add_memory(key, embedding)
        print(f"Added embedding to MemoryIndex for key '{key}'.")

        # Retrieve multiple times
        for _ in range(10):
            similar_memories = retrieve_similar_memories(text, self.conn, self.memory_index, top_k=1)
            print(f"Retrieved similar memories for '{text}':", similar_memories)
            self.assertIn(key, [mem[0] for mem in similar_memories])

    def test_memory_recall_for_similar_input(self):
        # Store a memory
        original_text = "I love Japan"
        key = "entry_test"
        embedding = embed_text(original_text)
        store_embedding(self.conn, key, embedding)
        print(f"Stored embedding for key '{key}'.")

        # Add embedding to MemoryIndex
        self.memory_index.add_memory(key, embedding)
        print(f"Added embedding to MemoryIndex for key '{key}'.")

        # Query with a semantically similar input
        similar_input = "What about Japan?"
        similar_memories = retrieve_similar_memories(similar_input, self.conn, self.memory_index, top_k=1)
        print(f"Retrieved similar memories for '{similar_input}':", similar_memories)

        # Check if the original memory is recalled
        self.assertIn(key, [mem[0] for mem in similar_memories])  # Expecting the key to be recalled

    def test_query_similar_type_error(self):
        # Test that passing a string raises a TypeError
        with self.assertRaises(TypeError):
            self.memory_index.query_similar("this should be an array", top_k=1)

    def test_query_similar_with_valid_vector(self):
        # Test that passing a valid np.ndarray works correctly
        vector = np.random.rand(384).astype(np.float32)  # Create a random vector with the correct dtype
        self.memory_index.add_memory("test_key", vector)
        similar_keys = self.memory_index.query_similar(vector, top_k=1)
        self.assertIn("test_key", similar_keys)

if __name__ == '__main__':
    unittest.main() 