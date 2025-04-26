import unittest
import asyncio
from prisma import Prisma
from prisma.errors import PrismaError

class TestDatabaseConnection(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Setup the database connection before running any tests."""
        cls.db = Prisma()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cls.db.connect())

    @classmethod
    def tearDownClass(cls):
        """Disconnect the database after all tests."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cls.db.disconnect())

    def test_database_connection(self):
        """Test that the program can connect to the database and perform a simple query."""
        loop = asyncio.get_event_loop()

        try:
            loop.run_until_complete(self.db.newstest.find_first())
            
            self.assertTrue(True, "Database connection and query successful.")
        except PrismaError as e:
            self.fail(f"Database connection failed: {e}")
        

if __name__ == '__main__':
    unittest.main()
