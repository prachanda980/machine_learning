import unittest
from src.predict import predict_email

class TestPredictEmail(unittest.TestCase):
    def test_predict_spam(self):
        result = predict_email("Win a free iPhone now!")
        self.assertIn(result, [0, 1])

if __name__ == "__main__":
    unittest.main()
