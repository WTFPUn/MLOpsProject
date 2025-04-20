import sys
import dotenv

import unittest

# from aws import TestS3Client
# from db import TestDatabaseConnection
from api import TestNewsAPI

# check dotenv file
dotenv_path = dotenv.find_dotenv()
if dotenv_path:
    print(f"Found .env file at {dotenv_path}")
    # export dotenv_path to environment variables
    dotenv.load_dotenv(dotenv_path)
    



if __name__ == '__main__':
    unittest.main()    

