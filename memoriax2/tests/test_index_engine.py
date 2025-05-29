import unittest
import numpy as np
from memoriax2.memory.index_engine import MemoryIndex

class TestMemoryIndex(unittest.TestCase):
   def setUp(self):
    self.embedding_dim = 128
    self.memory_index = MemoryIndex(self.embedding_dim)

    def test_add_and_query_memory(self):
        # Add a memory
        vector = [0.1] * 128  # Example 128-dimensional vector
        self.memory_index.add_memory('test_id', vector)

        # Query the memory
        result = self.memory_index.query_similar(vector, top_k=1)
        self.assertEqual(result, ['test_id'])

    def test_retrieval_accuracy(self):
        # Add multiple memories
        vector1 = [0.1] * 128
        vector2 = [0.2] * 128
        self.memory_index.add_memory('id1', vector1)
        self.memory_index.add_memory('id2', vector2)

        # Query similar to vector1
        result = self.memory_index.query_similar(vector1, top_k=1)
        self.assertEqual(result, ['id1'])

    def test_retrieval_speed(self):
        import time
        # Add a large number of memories
        for i in range(1000):
            vector = [i * 0.001] * 128
            self.memory_index.add_memory(f'id_{i}', vector)

        # Measure query time
        start_time = time.time()
        self.memory_index.query_similar([0.5] * 128, top_k=10)
        duration = time.time() - start_time

        # Assert that the query is fast
        self.assertLess(duration, 1)  # Query should take less than 1 second

if __name__ == '__main__':
    unittest.main() 