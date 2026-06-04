#!/usr/bin/env python3

import os
import sys
import time
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from personalized.gr_collaborative_filtering_model import NeuralCollaborativeFilteringModel
from personalized.gr_hybrid_model import NeuralHybridModel

class GRTopNPersonalized:
    def __init__(self, top_n=10):
        self.top_n = top_n
        self.batch_size = 2048
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model')
        self.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'output') 
        self.data_dir_personalized = os.path.join(self.data_dir, 'personalized')
        self.model_dir_personalized = os.path.join(self.model_dir, 'personalized')
        self.output_dir_personalized = os.path.join(self.output_dir, 'personalized') 
        self.cf_model_path = os.path.join(self.model_dir_personalized, 'gr_collaborative_filtering_model.pt')
        self.hybrid_model_path = os.path.join(self.model_dir_personalized, 'gr_hybrid_model.pt')
        self.top_n_output_path = os.path.join(self.output_dir_personalized, 'gr_top_n_personalized.json')
        self.min_user_ratings_count_threshold = 100
        self.min_user_ratings_std_dev_threshold = 1.25
        self.max_users_to_select = 1000

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'GRTopNPersonalized: using device {self.device}')

    def load(self):
        print('GRTopNPersonalized: loading data...')

        # load only needed columns from train
        print('GRTopNPersonalized: loading train reviews (minimal columns)...')
        train_minimal = pd.read_parquet(
            os.path.join(self.data_dir_personalized, 'gr_train_reviews.parquet'),
            columns=['user_id_csv', 'book_id_csv', 'rating']
        )
        print(f'GRTopNPersonalized: loaded train reviews successfully. shape={train_minimal.shape}')

        print('GRTopNPersonalized: loading test reviews (minimal columns)...')
        test_minimal = pd.read_parquet(
            os.path.join(self.data_dir_personalized, 'gr_test_reviews.parquet'),
            columns=['user_id_csv', 'book_id_csv']
        )
        print(f'GRTopNPersonalized: loaded test reviews successfully. shape={test_minimal.shape}')

        # precompute user stats
        print('GRTopNPersonalized: computing user stats...')
        self.user_stats = train_minimal.groupby('user_id_csv').agg(
            rating_count=('rating', 'count'),
            rating_std=('rating', 'std'),
            rating_mean=('rating', 'mean'),
        ).reset_index()
        print(f'GRTopNPersonalized: computed user stats successfully. num_users={len(self.user_stats)}')

        # precompute rated books per user
        print('GRTopNPersonalized: computing user rated books lookup...')
        self.user_rated_books = (
            pd.concat([
                train_minimal[['user_id_csv', 'book_id_csv']],
                test_minimal
            ])
            .groupby('user_id_csv')['book_id_csv']
            .apply(set)
            .to_dict()
        )
        print(f'GRTopNPersonalized: computed user rated books lookup successfully. num_users={len(self.user_rated_books)}')
        
        del train_minimal, test_minimal
        print('GRTopNPersonalized: freed train and test reviews from memory.')
    
        # load book metadata
        print('GRTopNPersonalized: loading book metadata (minimal columns)...')
        self.books_meta = pd.read_parquet(
            os.path.join(self.data_dir, 'gr_books.parquet'),
            columns=['book_id', 'title', 'authors']
        ).drop_duplicates('book_id').set_index('book_id')
        print(f'GRTopNPersonalized: loaded book metadata successfully. num_books={len(self.books_meta)}')

        # load content features
        print('GRTopNPersonalized: loading book content features...')
        self.book_content_features_df = pd.read_parquet(os.path.join(self.model_dir_personalized, 'gr_book_content_features.parquet'))
        print(f'GRTopNPersonalized: loaded book content features successfully. shape={self.book_content_features_df.shape}')

        # load primary authors
        print('GRTopNPersonalized: loading book primary authors...')
        self.book_primary_authors_df = pd.read_parquet(os.path.join(self.model_dir_personalized, 'gr_book_primary_authors.parquet'))
        print(f'GRTopNPersonalized: loaded book primary authors successfully. shape={self.book_primary_authors_df.shape}')

        # load id maps
        print('GRTopNPersonalized: loading user and book id maps...')
        self.user_id_map_df = pd.read_csv(os.path.join(self.data_dir, 'gr_user_id_map.csv'))
        self.book_id_map_df = pd.read_csv(os.path.join(self.data_dir, 'gr_book_id_map.csv'))
        print(f'GRTopNPersonalized: loaded id maps successfully. num_users={len(self.user_id_map_df)}, num_books={len(self.book_id_map_df)}')

        self.num_users = len(self.user_id_map_df)
        self.num_books = len(self.book_id_map_df)
        self.num_authors = int(self.book_primary_authors_df['author_id_csv'].max())
        self.max_authors = int(self.book_primary_authors_df.groupby('book_id_csv')['author_id_csv'].count().max())
        self.content_feature_dim = len([c for c in self.book_content_features_df.columns
                                        if c.startswith('genre_') or c.startswith('lang_') or
                                        c in ['publication_year', 'weighted_compound_sentiment_score_avg',
                                            'rating_mean', 'review_count']])

        # precompute author lookup
        print('GRTopNPersonalized: precomputing author lookup...')
        authors_grouped = self.book_primary_authors_df.groupby('book_id_csv')['author_id_csv'].apply(list)
        self.book_authors_lookup = {}
        for book_id_csv, author_list in authors_grouped.items():
            padded = author_list[:self.max_authors]
            padded = padded + [0] * (self.max_authors - len(padded))
            self.book_authors_lookup[book_id_csv] = padded
        del self.book_primary_authors_df
        print(f'GRTopNPersonalized: precomputed author lookup successfully. num_authors={len(self.book_authors_lookup)}')

        # precompute content features lookup
        print('GRTopNPersonalized: indexing content features...')
        content_cols = [c for c in self.book_content_features_df.columns
                        if c.startswith('genre_') or c.startswith('lang_') or
                        c in ['publication_year', 'weighted_compound_sentiment_score_avg',
                            'rating_mean', 'review_count']]
        self.book_content_features_df = self.book_content_features_df.set_index('book_id_csv')
        self.book_content_features_df = self.book_content_features_df[self.book_content_features_df.index.notna()]
        print(f'GRTopNPersonalized: dropped NaN book ids. num_books={len(self.book_content_features_df)}')
        self.content_cols = content_cols
        print(f'GRTopNPersonalized: indexed content features successfully. content_feature_dim={self.content_feature_dim}')

        # precompute book metadata lookup
        print('GRTopNPersonalized: precomputing book id csv to book id lookup...')
        self.book_id_csv_to_book_id = self.book_id_map_df.set_index('book_id_csv')['book_id']
        print('GRTopNPersonalized: precomputed book id csv to book id lookup successfully.')

        # precompute author and content tensors for all books
        print('GRTopNPersonalized: precomputing author and content tensors for all books...')
        valid_book_ids = set().union(*self.user_rated_books.values())
        print(f'GRTopNPersonalized: valid book ids: {len(valid_book_ids)}')
        self.all_book_ids = [b for b in self.book_content_features_df.index.tolist() if b in valid_book_ids]
        print(f'GRTopNPersonalized: filtered to {len(self.all_book_ids)} valid books.')
        self.all_book_id_to_idx = {b: i for i, b in enumerate(self.all_book_ids)}

        print('GRTopNPersonalized: precomputing content tensor (vectorized)...')
        content_array = self.book_content_features_df.loc[self.all_book_ids, content_cols].values.astype(np.float32)
        self.all_content_tensor = torch.tensor(content_array, dtype=torch.float32)

        del content_array # free intermediate numpy array

        print(f'GRTopNPersonalized: content tensor done. shape={self.all_content_tensor.shape}')

        print('GRTopNPersonalized: precomputing author tensor (vectorized)...')

        author_array = np.array([
            self.book_authors_lookup.get(b, [0] * self.max_authors)
            for b in self.all_book_ids
        ], dtype=np.int64)
        self.all_author_tensor = torch.tensor(author_array, dtype=torch.long)

        del author_array  # free intermediate numpy array

        print(f'GRTopNPersonalized: author tensor done. shape={self.all_author_tensor.shape}')

        del self.book_content_features_df
        del self.book_authors_lookup

        print(f'GRTopNPersonalized: loaded data successfully.')
        print(f'GRTopNPersonalized: num_users={self.num_users}, num_books={self.num_books}')
        print(f'GRTopNPersonalized: num_authors={self.num_authors}, max_authors={self.max_authors}')
        print(f'GRTopNPersonalized: content_feature_dim={self.content_feature_dim}')

    def select_users(self):
        print(f'GRTopNPersonalized: selecting top {self.max_users_to_select} users min_user_ratings_count_threshold={self.min_user_ratings_count_threshold}, min_user_ratings_std_dev_threshold={self.min_user_ratings_std_dev_threshold}')
        qualified = self.user_stats[
            (self.user_stats['rating_count'] >= self.min_user_ratings_count_threshold) &
            (self.user_stats['rating_std'] >= self.min_user_ratings_std_dev_threshold)
        ].sort_values('rating_count', ascending=False).head(self.max_users_to_select)

        self.sample_user_ids = qualified['user_id_csv'].tolist()
        print(f'GRTopNPersonalized: selected {len(self.sample_user_ids)} users.')
        return qualified

    def _load_cf_model(self):
        embedding_dim = 64
        checkpoint = torch.load(self.cf_model_path, map_location=self.device)
        num_users = checkpoint['user_embedding.weight'].shape[0]
        num_books = checkpoint['book_embedding.weight'].shape[0]
        cf_model = NeuralCollaborativeFilteringModel(
            num_users, num_books, embedding_dim
        ).to(self.device)
        cf_model.load_state_dict(checkpoint)
        cf_model.eval()
        print('GRTopNPersonalized: loaded CF model.')
        return cf_model

    def _load_hybrid_model(self):
        embedding_dim = 64
        checkpoint = torch.load(self.hybrid_model_path, map_location=self.device)
        num_users = checkpoint['user_embedding.weight'].shape[0]
        num_books = checkpoint['book_embedding.weight'].shape[0]
        hybrid_model = NeuralHybridModel(
            num_users, num_books, self.num_authors,
            self.content_feature_dim, embedding_dim
        ).to(self.device)
        hybrid_model.load_state_dict(checkpoint)
        hybrid_model.eval()
        print('GRTopNPersonalized: loaded Hybrid model.')
        return hybrid_model

    def _get_unrated_books(self, user_id_csv):
        rated_books = self.user_rated_books.get(user_id_csv, set())
        return [b for b in self.all_book_ids if b not in rated_books]

    def _predict_cf(self, cf_model, user_id_csv, book_ids):
        all_predictions = []
        user_tensor = torch.tensor([user_id_csv] * len(book_ids), dtype=torch.long)
        book_tensor = torch.tensor(book_ids, dtype=torch.long)

        for i in range(0, len(book_ids), self.batch_size):
            user_batch = user_tensor[i:i+self.batch_size].to(self.device)
            book_batch = book_tensor[i:i+self.batch_size].to(self.device)
            with torch.no_grad():
                preds = cf_model(user_batch, book_batch)
            all_predictions.extend(preds.cpu().numpy().tolist())

        return all_predictions

    def _predict_hybrid(self, hybrid_model, user_id_csv, book_ids):
        indices = [self.all_book_id_to_idx[b] for b in book_ids]
        all_predictions = []
        user_tensor = torch.tensor([user_id_csv] * len(book_ids), dtype=torch.long)
        book_tensor = torch.tensor(book_ids, dtype=torch.long)
        author_tensor = self.all_author_tensor[indices]
        content_tensor = self.all_content_tensor[indices]

        for i in range(0, len(book_ids), self.batch_size):
            user_batch = user_tensor[i:i+self.batch_size].to(self.device)
            book_batch = book_tensor[i:i+self.batch_size].to(self.device)
            author_batch = author_tensor[i:i+self.batch_size].to(self.device)
            content_batch = content_tensor[i:i+self.batch_size].to(self.device)
            with torch.no_grad():
                preds = hybrid_model(user_batch, book_batch, author_batch, content_batch)
            all_predictions.extend(preds.cpu().numpy().tolist())

        return all_predictions

    def _get_top_n_books(self, book_ids, predictions):
        sorted_books = sorted(zip(book_ids, predictions), key=lambda x: x[1], reverse=True)
        return sorted_books[:self.top_n]

    def generate_top_n(self):
        print('GRTopNPersonalized: generating Top-N recommendations...')
        cf_model = self._load_cf_model()
        hybrid_model = self._load_hybrid_model()

        def enrich(top_n):
            enriched = []
            for book_id_csv, predicted_rating in top_n:
                book_id = self.book_id_csv_to_book_id.get(book_id_csv)
                if book_id in self.books_meta.index:
                    title = self.books_meta.loc[book_id, 'title']
                    authors = self.books_meta.loc[book_id, 'authors']
                else:
                    title = 'Unknown'
                    authors = 'Unknown'
                enriched.append({
                    'book_id_csv': book_id_csv,
                    'title': title,
                    'authors': authors,
                    'predicted_rating': predicted_rating
                })
            return enriched

        rows = []
        start_time = time.time()
        for i, user_id_csv in enumerate(self.sample_user_ids):
            if i % 100 == 0 and i > 0:
                elapsed = time.time() - start_time
                per_user = elapsed / i
                remaining = per_user * (len(self.sample_user_ids) - i)
                print(f'GRTopNPersonalized: processing user {i+1}/{len(self.sample_user_ids)}... elapsed={elapsed:.0f}s, per_user={per_user:.1f}s, eta={remaining:.0f}s')
            elif i == 0:
                print(f'GRTopNPersonalized: processing user {i+1}/{len(self.sample_user_ids)}...')

            unrated_books = self._get_unrated_books(user_id_csv)

            # CF predictions and top N
            cf_preds = self._predict_cf(cf_model, user_id_csv, unrated_books)
            cf_top_n = self._get_top_n_books(unrated_books, cf_preds)

            # Hybrid predictions and top N
            hybrid_preds = self._predict_hybrid(hybrid_model, user_id_csv, unrated_books)
            hybrid_top_n = self._get_top_n_books(unrated_books, hybrid_preds)

            # enrich both lists with title and authors
            cf_enriched = enrich(cf_top_n)
            hybrid_enriched = enrich(hybrid_top_n)

            # compute intersection by reusing enriched lists
            cf_books = {item['book_id_csv']: item for item in cf_enriched}
            hybrid_books = {item['book_id_csv']: item for item in hybrid_enriched}
            intersection_ids = set(cf_books.keys()) & set(hybrid_books.keys())

            # extract only book_id_csv and title for intersection
            intersection_books = [
                {
                    'book_id_csv': book_id_csv,
                    'title': cf_books[book_id_csv]['title']
                }
                for book_id_csv in intersection_ids
            ]

            rows.append({
                'user_id_csv': user_id_csv,
                'cf_recommendations': cf_enriched,
                'hybrid_recommendations': hybrid_enriched,
                'intersection_count': len(intersection_ids),
                'intersection_books': intersection_books
            })

        self.recommendations_df = pd.DataFrame(rows)
        print(f'GRTopNPersonalized: generated {len(self.recommendations_df)} recommendations.')

    def save(self):
        print('GRTopNPersonalized: saving recommendations...')

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.integer)):
                    return int(obj)
                if isinstance(obj, (np.floating)):
                    return float(obj)
                return super().default(obj)        

        with open(self.top_n_output_path, 'w') as f:
            for record in self.recommendations_df.to_dict(orient='records'):
                f.write(json.dumps(record, cls=NumpyEncoder) + '\n')
        print(f'GRTopNPersonalized: saved to {self.top_n_output_path}')

    def display(self):
        print('\n========== GRTopNPersonalized Results ==========')
        with open(self.top_n_output_path, 'r') as f:
            records = [json.loads(line) for line in f]
        print(f'Total users: {len(records)}')
        intersection_counts = [r['intersection_count'] for r in records]
        print(f'Intersection Mean: {np.mean(intersection_counts):.2f} (min={min(intersection_counts)}, max={max(intersection_counts)})')
        print('\n--- First 5 Users ---')
        for rec in records[:5]:
            print(f'\n  user_id_csv={rec["user_id_csv"]}')
            print(f'  CF Top 10:')
            for i, book in enumerate(rec['cf_recommendations']):
                print(f'    {i+1}. {book["title"]} — {book["authors"]} (predicted={book["predicted_rating"]:.2f})')
            print(f'  Hybrid Top 10:')
            for i, book in enumerate(rec['hybrid_recommendations']):
                print(f'    {i+1}. {book["title"]} — {book["authors"]} (predicted={book["predicted_rating"]:.2f})')
            print(f'  Intersection ({rec["intersection_count"]} books):')
            for book in rec['intersection_books']:
                print(f'    - {book["title"]}')
        print(f'==================== Refer to {self.top_n_output_path} for all the results =============================\n')

    def run(self, override=False):
        print(f'GRTopNPersonalized: starting with override={override}')
        if not override and os.path.exists(self.top_n_output_path):
            print('GRTopNPersonalized: output already exists, skipping. Use override=True to regenerate.')
            self.display()
            return
        self.load()
        self.select_users()
        self.generate_top_n()
        self.save()
        self.display()
        print('GRTopNPersonalized: done.')