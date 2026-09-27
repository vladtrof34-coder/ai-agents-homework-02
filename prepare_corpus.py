"""Повторное извлечение корпуса: python prepare_corpus.py путь/к/книге.pdf"""
import argparse
import json
from pathlib import Path
from pypdf import PdfReader

parser = argparse.ArgumentParser()
parser.add_argument('pdf', type=Path)
args = parser.parse_args()
rows = []
for number, page in enumerate(PdfReader(args.pdf).pages, 1):
    text = ' '.join((page.extract_text() or '').split())
    if len(text) >= 80:
        rows.append({'page': args.pdf.name, 'pdf_page': number,
                     'section': f'страница PDF {number}', 'url': '', 'text': text})
output = Path(__file__).resolve().parent / 'data' / 'corpus.jsonl'
output.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
print(f'{len(rows)} страниц сохранено в {output}')
