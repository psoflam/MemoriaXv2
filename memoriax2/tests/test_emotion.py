import unittest
from memoriax2.nlp.emotion import detect_emotion
from memoriax2.storage.database import store_in_db, init_db
from memoriax2.core.chatbot import generate_response

class TestEmotionDetection(unittest.TestCase):

    def test_detect_emotion(self):
        self.assertEqual(detect_emotion("I am so happy today!"), "happy")
        self.assertEqual(detect_emotion("This is terrible, I'm so angry!"), "angry")
        self.assertEqual(detect_emotion("I feel a bit down."), "sad")
        self.assertEqual(detect_emotion("It's just an ordinary day."), "neutral")

class TestDatabaseStorage(unittest.TestCase):

    def setUp(self):
        self.conn = init_db()

    def test_store_in_db(self):
        session_id = "test_session"
        memory_key = "test_key"
        user_input = "I am happy"
        response = "That's great to hear!"
        emotion = "happy"
        store_in_db(self.conn, session_id, memory_key, user_input, response, emotion)
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM session_messages WHERE session_id=? AND user_input=?", (session_id, memory_key))
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        self.assertEqual(result[0], session_id)
        self.assertEqual(result[1], memory_key)
        self.assertEqual(result[2], response)
        self.assertEqual(result[3], emotion)

class TestResponseGeneration(unittest.TestCase):

    def test_generate_response(self):
        response = generate_response("I am so sad.")
        self.assertIn("empathetic", response)  # Assuming the response contains some empathetic phrasing

if __name__ == '__main__':
    unittest.main() 