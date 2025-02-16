from unittest import TestCase

class Test(TestCase):
    def test_get_db(self):
        from sdl_orchestration.database import get_db
        db = get_db()
        self.assertIsNotNone(db)
