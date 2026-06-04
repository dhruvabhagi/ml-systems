#!/usr/bin/env python3

import os
import json
import pandas as pd
import numpy as np
from gr_book_metadata_handler import BookMetadataHandler
from gr_reviews_sentiment_analyzer import GoodreadsReviewsSentimentAnalyzer


class GRTopNGlobal:
    def __init__(self, min_reviews=50, top_n=20):
        self.min_reviews = min_reviews
        self.top_n = top_n
        self.goodreads_reviews_sentiment_analyzer_score_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'gr_reviews_sentiment_score.parquet')
        self.output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'gr_top_n_global.json')
        self.book_metadata_handler = BookMetadataHandler()

    def load(self):
        print(f'GRTopNGlobal: loading review sentiment scores from the parquet file {self.goodreads_reviews_sentiment_analyzer_score_path}')
        self.book_review_sentiment = pd.read_parquet(self.goodreads_reviews_sentiment_analyzer_score_path)
        self.book_metadata_handler.load()
        self.book_metadata_handler.preprocess()
        print('GRTopNGlobal: loaded successfully.')

    def filter(self):
        print(f'GRTopNGlobal: filtering to books with min_reviews={self.min_reviews}...')
        self.filtered = self.book_review_sentiment[
            self.book_review_sentiment['review_count'] >= self.min_reviews
        ]
        print(f'GRTopNGlobal: {len(self.filtered)} books after filtering.')

    def generate_top_n(self):
        print(f'GRTopNGlobal: generating top {self.top_n} books...')
        top = self.filtered.sort_values('weighted_compound_sentiment_score_avg', ascending=False).head(self.top_n)
        self.top_books = pd.merge(
            top,
            self.book_metadata_handler.get()[['book_id', 'title', 'authors', 'genres', 'language_code']],
            on='book_id',
            how='left'
        )
        print(f'GRTopNGlobal: generated top {self.top_n} books.')

    def display(self):
        print('\n========== GRTopNGlobal Results ==========')
        with open(self.output_path, 'r') as f:
            records = [json.loads(line) for line in f]
        print(f'Total books: {len(records)}')
        print('\n--- Top 20 Global Books ---')
        for i, rec in enumerate(records):
            print(f'  {i+1}. {rec["title"]} — {rec["authors"]} (score={rec["weighted_compound_sentiment_score_avg"]:.4f}, reviews={int(rec["review_count"])})')
        print('==========================================\n')

    def save(self):
        print(f'GRTopNGlobal: saving to {self.output_path}...')
        with open(self.output_path, 'w') as f:
            for record in self.top_books.to_dict(orient='records'):
                # convert any numpy arrays to lists for JSON serialization
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        record[key] = value.tolist()
                f.write(json.dumps(record) + '\n')
        print('GRTopNGlobal: saved successfully.')

    def run(self, override=False):
        print(f'GRTopNGlobal: starting with override={override}')
        if not override and os.path.exists(self.output_path):
            print('GRTopNGlobal: output already exists, skipping. Use override=True to regenerate.')
            self.display()
            return
        self.load()
        self.filter()
        self.generate_top_n()
        self.display()
        self.save()
        print('GRTopNGlobal: done.')