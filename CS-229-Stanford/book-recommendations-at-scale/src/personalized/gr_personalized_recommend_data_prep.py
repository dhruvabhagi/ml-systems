#!/usr/bin/env python3

import os
import sys
import pandas as pd

# add src to path so we can import from parent directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from gr_review_metadata_handler import ReviewMetadataHandler #type: ignore
from gr_book_metadata_handler import BookMetadataHandler #type: ignore

class GRPersonalizedRecommendDataPrep:
    def __init__(self):
        self.review_metadata_handler = ReviewMetadataHandler()
        self.book_metadata_handler = BookMetadataHandler()
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model') 
        self.data_dir_personalized = os.path.join(self.data_dir, 'personalized')
        self.model_dir_personalized = os.path.join(self.model_dir, 'personalized')
        self.train_reviews_path = os.path.join(self.data_dir_personalized, 'gr_train_reviews.parquet')
        self.validation_reviews_path = os.path.join(self.data_dir_personalized, 'gr_validation_reviews.parquet')
        self.test_reviews_path = os.path.join(self.data_dir_personalized, 'gr_test_reviews.parquet')
        self.content_features_path = os.path.join(self.model_dir_personalized, 'gr_content_filtering_features.parquet')
        self.sentiment_score_path = os.path.join(self.model_dir, 'gr_reviews_sentiment_score.parquet')

    def load_and_preprocess(self):
        print('GRPersonalizedRecommendDataPrep: loading data...')
        self.book_metadata_handler.load()
        self.book_metadata_handler.preprocess()
        self.review_metadata_handler.load()
        self.review_metadata_handler.preprocess()
        self.book_metadata = self.book_metadata_handler.get()
        self.reviews = self.review_metadata_handler.get()
        self.sentiment_scores = pd.read_parquet(self.sentiment_score_path)
         # get book_id_csv from book_metadata and merge into reviews
        # book_id_csv belongs to BookMetadataHandler by design
        book_id_mapping = self.book_metadata[['book_id', 'book_id_csv']]
        self.reviews = pd.merge(self.reviews, book_id_mapping, on='book_id', how='left')
        print(f'GRPersonalizedRecommendDataPrep: Null book_id_csv in reviews: {self.reviews["book_id_csv"].isna().sum()}')
        print('GRPersonalizedRecommendDataPrep: loading data completed.')

    def filter(self, min_user_ratings=10, min_book_ratings=25):
        print('GRPersonalizedRecommendDataPrep: filtering active users and popular books...')
        user_counts = self.reviews.groupby('user_id')['rating'].count()
        active_users = user_counts[user_counts >= min_user_ratings].index
        book_counts = self.reviews.groupby('book_id')['rating'].count()
        popular_books = book_counts[book_counts >= min_book_ratings].index
        self.reviews = self.reviews[
            self.reviews['user_id'].isin(active_users) &
            self.reviews['book_id'].isin(popular_books)
        ]
        print(f'GRPersonalizedRecommendDataPrep: filtered to {len(self.reviews)} reviews, {len(active_users)} users, {len(popular_books)} books.')
 
    def split(self):
        print('GRPersonalizedRecommendDataPrep: splitting data into train/val/test...')
        self.reviews['read_at'] = pd.to_datetime(self.reviews['read_at'], errors='coerce')

        # check null read_at before splitting
        print(f"GRPersonalizedRecommendDataPrep: Total reviews: {len(self.reviews)}")
        print(f"GRPersonalizedRecommendDataPrep: Null read_at: {self.reviews['read_at'].isna().sum()}")
        print(f"GRPersonalizedRecommendDataPrep: Null read_at %: {self.reviews['read_at'].isna().mean() * 100:.1f}%")

        self.reviews = self.reviews.sort_values(['user_id', 'read_at'], na_position='first')

        train_reviews_list, val_reviews_list, test_reviews_list = [], [], []
        for _, user_reviews in self.reviews.groupby('user_id'):
            n = len(user_reviews)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)
            train_reviews_list.append(user_reviews.iloc[:train_end])
            val_reviews_list.append(user_reviews.iloc[train_end:val_end])
            test_reviews_list.append(user_reviews.iloc[val_end:])

        self.train_reviews = pd.concat(train_reviews_list, ignore_index=True)
        self.validation_reviews = pd.concat(val_reviews_list, ignore_index=True)
        self.test_reviews = pd.concat(test_reviews_list, ignore_index=True)
        print(f'GRPersonalizedRecommendDataPrep: train={len(self.train_reviews)}, val={len(self.validation_reviews)}, test={len(self.test_reviews)}')

    def build_content_filtering_features(self): # content-based filtering features
        print('GRPersonalizedRecommendDataPrep: building content based filtering features...')
        self.content_features = pd.merge(
            self.book_metadata[['book_id', 'genres', 'book_id_csv', 'language_code', 'authors', 'publication_year']],
            self.sentiment_scores[['book_id', 'weighted_compound_sentiment_score_avg', 'rating_mean', 'review_count']],
            on='book_id',
            how='left'
        ) 
        print('GRPersonalizedRecommendDataPrep: successfully built content based filtering features.')

    def save(self):
        print('GRPersonalizedRecommendDataPrep: saving train_reviews, validation_reviews, test_reviews and content-based filtering features to parquet...')
        # drop read_at before saving to avoid string to datetime conversion issues, especially if there are NA values in read_at,
        # which there are (around 170), as we saw in the earlier print statements
        self.train_reviews = self.train_reviews.drop(columns=['read_at'])
        self.validation_reviews = self.validation_reviews.drop(columns=['read_at'])
        self.test_reviews = self.test_reviews.drop(columns=['read_at'])
        # to_parquet overrides existing files if they exist, so no need to check for existence before saving
        self.train_reviews.to_parquet(self.train_reviews_path, index=False)
        self.validation_reviews.to_parquet(self.validation_reviews_path, index=False)
        self.test_reviews.to_parquet(self.test_reviews_path, index=False)
        self.content_features.to_parquet(self.content_features_path, index=False)
        print('GRPersonalizedRecommendDataPrep: successfully saved all data.')

    def run(self, override=False):
        print(f'GRPersonalizedRecommendDataPrep: starting data preparation process (run) with override={override}')
        if not override and os.path.exists(self.train_reviews_path) and os.path.exists(self.validation_reviews_path) and os.path.exists(self.test_reviews_path) and os.path.exists(self.content_features_path):
            print('GRPersonalizedRecommendDataPrep: run is loading existing train_reviews, validation_reviews, test_reviews and content_features from parquet...')
            self.train_reviews = pd.read_parquet(self.train_reviews_path)
            self.validation_reviews = pd.read_parquet(self.validation_reviews_path)
            self.test_reviews = pd.read_parquet(self.test_reviews_path)
            self.content_features = pd.read_parquet(self.content_features_path)
            print('GRPersonalizedRecommendDataPrep: loaded existing train_reviews, validation_reviews, test_reviews and content_features successfully.')
            return
        print('GRPersonalizedRecommendDataPrep: no existing data found or override=True, starting full data preparation process (run)...') 
        self.load_and_preprocess()
        self.filter()
        self.split()
        self.build_cf_features()
        self.save()
        print('GRPersonalizedRecommendDataPrep: successfully completed full data preparation process (run)')