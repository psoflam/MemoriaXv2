import unittest
from unittest.mock import MagicMock
from memoriax2.core.chatbot import process_user_input, detect_emotion

class TestMemoryAutosave(unittest.TestCase):
    def setUp(self):
        """Set up mock database connection and memory index for tests."""
        self.conn = MagicMock()
        self.memory_index = MagicMock()
        self.session_id = 'test_session'

    def test_name_detection(self):
        """Test if the system detects a name in the user input."""
        user_input = "My name is Colin"
        response, emotion = process_user_input(user_input, self.conn, self.session_id, self.memory_index)
        self.assertIn("name", response)

    def test_goal_detection(self):
        """Test if the system detects a goal in the user input."""
        user_input = "I want to move to Japan"
        response, emotion = process_user_input(user_input, self.conn, self.session_id, self.memory_index)
        self.assertIn("goal", response)

    def test_emotional_prompt(self):
        """Test if the system detects an emotional trigger in the user input."""
        user_input = "I feel lost"
        response, emotion = process_user_input(user_input, self.conn, self.session_id, self.memory_index)
        self.assertIn("emotional trigger", response)

    def test_memory_deletion_confirmation(self):
        """Test if the system asks for confirmation before deleting a memory."""
        user_input = "Forget that I said I want to move to Japan"
        response, emotion = process_user_input(user_input, self.conn, self.session_id, self.memory_index)
        self.assertIn("Are you sure you want to forget", response)

    def test_emotion_tagging_on_sad_messages(self):
        """Test if the system correctly tags sad emotions."""
        user_input = "I am sad"
        emotion = detect_emotion(user_input)
        self.assertEqual(emotion, "sad")

if __name__ == '__main__':
    unittest.main() 