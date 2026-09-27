import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import homework02 as hw


class ToolTests(unittest.TestCase):
    def test_found(self):
        hits = [{'text': 'Один вариант в день', 'pdf_page': 29}]
        with patch.object(hw, 'search_milvus', return_value=hits):
            self.assertEqual(hw.knowledge_base('сколько вариантов?'), hits)

    def test_empty(self):
        with patch.object(hw, 'search_milvus', return_value=[]):
            self.assertEqual(hw.knowledge_base('нет ответа'), [])

    def test_filter(self):
        with patch.object(hw, 'search_milvus', return_value=[]) as search:
            hw.knowledge_base('проверка', 'book.pdf')
            search.assert_called_once_with('проверка', page='book.pdf')


class FakeDB:
    def __init__(self): self.rows = []
    def has_collection(self, name): return bool(self.rows)
    def drop_collection(self, name): self.rows = []
    def create_collection(self, *args, **kwargs): pass
    def insert(self, name, rows): self.rows = rows
    def flush(self, name): pass
    def search(self, name, data, limit, output_fields):
        vectors = np.array([r['vector'] for r in self.rows])
        top = np.argsort(-(vectors @ np.array(data[0])))[:limit]
        return [[{'entity': {'text': self.rows[i]['text']}} for i in top]]


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        def embed(texts):
            return np.array([[float('city' in t or 'город' in t), float('name' in t)] for t in texts])
        self.memory = hw.Memory(FakeDB(), embed, Path(self.tmp.name) / 'facts.json')
    def tearDown(self): self.tmp.cleanup()

    def test_empty(self):
        self.memory.store({})
        self.assertEqual(self.memory.recall('город'), [])

    def test_found(self):
        self.memory.store({'city': 'Казань', 'name': 'Лена'})
        self.assertEqual(self.memory.recall('город', 1), ['city: Казань'])

    def test_replaced(self):
        self.memory.store({'city': 'Казань'})
        self.memory.store({'city': 'Пермь'})
        self.assertEqual(self.memory.recall('город'), ['city: Пермь'])
        self.assertEqual(self.memory.known(), {'city': 'Пермь'})
        self.memory.store({})
        self.assertEqual(self.memory.recall('город'), [])


if __name__ == '__main__':
    unittest.main()
