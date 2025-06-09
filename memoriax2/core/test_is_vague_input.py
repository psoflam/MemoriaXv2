import unittest
from memoriax2.core.chatbot import is_vague_input


class TestIsVagueInput(unittest.TestCase):
    def test_vague_inputs(self):
        # Test cases for vague inputs
        self.assertTrue(is_vague_input("ok."))
        self.assertTrue(is_vague_input("cool"))
        self.assertTrue(is_vague_input("sure"))
        self.assertTrue(is_vague_input("yeah"))
        self.assertTrue(is_vague_input(" "))
        self.assertTrue(is_vague_input("..."))

    def test_non_vague_inputs(self):
        # Test cases for non-vague inputs
        self.assertFalse(is_vague_input("What do you remember about me?"))
        self.assertFalse(is_vague_input("Tell me about last time."))
        self.assertFalse(is_vague_input("Well I'm here to talk to you."))
        self.assertFalse(is_vague_input("Do you remember our last conversation?"))


if __name__ == '__main__':
    unittest.main() 