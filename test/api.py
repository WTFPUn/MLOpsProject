import unittest
import requests
from prisma.models import NewsTest

class TestNewsAPI(unittest.TestCase):

    BASE_URL = "http://localhost:8000"  # URL where your FastAPI app is running

    # def test_get_news_no_date(self):
    #     """Test case when no date is provided"""
    #     response = requests.get(f"{self.BASE_URL}/news?test=true")
    #     self.assertEqual(response.status_code, 200)
    #     try:
    #         result = NewsTest(**response.json())
    #         print(result)
    #     except Exception as e:
    #         self.fail(f"Failed to parse response: {e}")

    def test_get_news_with_date(self):
        """Test case when a specific date is provided"""
        date_param = "2022-01-01"
        
        response = requests.get(f"{self.BASE_URL}/news?date={date_param}&test=true")
        self.assertEqual(response.status_code, 200)
        try:
            result = NewsTest(**response.json())
            print(result)
        except Exception as e:
            self.fail(f"Failed to parse response: {e}")

    # def test_get_news_no_results(self):
    #     """Test case when no news is found for the given date"""
    #     date_param = "9999-12-31"  # Use a date that would likely return no results
    #     response = requests.get(f"{self.BASE_URL}/news?date={date_param}&test=true")
    #     print("=============")
    #     print(response.json())
    #     self.assertEqual(response.status_code, 404)
       


if __name__ == "__main__":
    unittest.main()
