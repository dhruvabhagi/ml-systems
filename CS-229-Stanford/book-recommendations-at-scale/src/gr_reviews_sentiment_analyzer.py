#!/usr/bin/env python3

import os
import pandas as pd
import nltk
import tqdm
from multiprocessing import Pool
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gr_review_metadata_handler import ReviewMetadataHandler

def _init_worker():
    global sia
    nltk.download('vader_lexicon', quiet=True)
    sia = SentimentIntensityAnalyzer()

def _score_text(text):
    return sia.polarity_scores(text)['compound']

class GoodreadsReviewsSentimentAnalyzer:
    def __init__(self):
        self.review_metadata_handler = ReviewMetadataHandler()
        self.goodreads_reviews_sentiment_analyzer_score_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'model', 'gr_reviews_sentiment_score.parquet')

    def load(self):
        self.review_metadata_handler.load()

    def preprocess(self):
        self.review_metadata_handler.preprocess()

    def score(self):
        print('GoodreadsReviewsSentimentAnalyzer: scoring reviews with VADER sentiment analyzer...')
        aggregated_chunks = []
        with Pool(initializer=_init_worker) as pool:
            with tqdm.tqdm() as pbar:
                for chunk in self.review_metadata_handler.iter_reviews_for_scoring(
                    batch_size=500_000
                ):
                    chunk = chunk.dropna(subset=['review_text', 'book_id'])
                    chunk['compound_sentiment_score'] = pool.map(_score_text, chunk['review_text'])
                    chunk = chunk.drop(columns=['review_text'])
                    chunk['vote'] = chunk['n_votes'] + 1
                    chunk['weighted_compound_sentiment_score'] = chunk['compound_sentiment_score'] * chunk['vote']
                    agg = (
                        chunk.groupby('book_id')
                        .agg(
                            compound_sentiment_score_sum=('compound_sentiment_score', 'sum'),
                            weighted_compound_sentiment_score_sum=('weighted_compound_sentiment_score', 'sum'),
                            total_vote_sum=('vote', 'sum'),
                            review_count=('compound_sentiment_score', 'count'),
                            rating_sum=('rating', 'sum'),
                        )
                        .reset_index()
                    )
                    aggregated_chunks.append(agg)
                    pbar.update(len(chunk))

        print('GoodreadsReviewsSentimentAnalyzer: combining aggregated chunks...')
        combined = pd.concat(aggregated_chunks).groupby('book_id').agg(
            compound_sentiment_score_sum=('compound_sentiment_score_sum', 'sum'),
            weighted_compound_sentiment_score_sum=('weighted_compound_sentiment_score_sum', 'sum'),
            total_vote_sum=('total_vote_sum', 'sum'),
            review_count=('review_count', 'sum'),
            rating_sum=('rating_sum', 'sum'),
        ).reset_index()

        # compute means at the end from sums — avoids mean of means bug
        # | review  | compound_sentiment_score | vote | weighted_compound_sentiment_score | rating | book_id |
        # | review1 |           0.5            |  10  |                5                  |   4    |   1     |
        # | review2 |           0.2            |   5  |                1                  |   3    |   1     |
        # | book_id | compound_sentiment_score_mean | weighted_compound_sentiment_score_sum | total_vote_sum | review_count | rating_mean | weighted_compound_sentiment_score_avg |
        # |   1     |             0.35              |                6                      |       15       |      2       |     3.5     |         6/15 = 0.4                    |
        combined['rating_mean'] = combined['rating_sum'] / combined['review_count']
        combined['compound_sentiment_score_mean'] = combined['compound_sentiment_score_sum'] / combined['review_count']
        combined['weighted_compound_sentiment_score_avg'] = (
            combined['weighted_compound_sentiment_score_sum'] / combined['total_vote_sum']
        )
        self.book_review_sentiment = combined
        print('GoodreadsReviewsSentimentAnalyzer: scoring complete, saving to parquet...')
        self.book_review_sentiment.to_parquet(self.goodreads_reviews_sentiment_analyzer_score_path, index=False)
        print('GoodreadsReviewsSentimentAnalyzer: successfully saved scored reviews to parquet.')

    def run(self, override=False):
        print(f'GoodreadsReviewsSentimentAnalyzer: starting with override={override}')
        if not override and os.path.exists(self.goodreads_reviews_sentiment_analyzer_score_path):
            print('GoodreadsReviewsSentimentAnalyzer: output already exists, skipping. Use override=True to regenerate.')
            return
        self.load()
        self.preprocess()
        self.score()