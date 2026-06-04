#!/usr/bin/env python3

import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

GENRE_KEYS = [
    'children', 'comics, graphic', 'fantasy, paranormal', 'fiction',
    'history, historical fiction, biography', 'mystery, thriller, crime',
    'non-fiction', 'poetry', 'romance', 'young-adult'
]

LANGUAGE_MAP = {
    'eng': 'english', 'en-US': 'english', 'en-GB': 'english',
    'en-CA': 'english', 'en-IN': 'english', 'en': 'english',
    'spa': 'spanish', 'es-MX': 'spanish',
    'ger': 'german',
    'fre': 'french',
    'por': 'portuguese', 'pt-BR': 'portuguese'
}

LANGUAGE_CATEGORIES = ['english', 'spanish', 'german', 'french', 'portuguese', 'other']

class GRBookContentFeatureBuilder:
    def __init__(self):
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model')
        self.model_dir_personalized = os.path.join(self.model_dir, 'personalized')
        self.content_features_path_input = os.path.join(self.model_dir_personalized, 'gr_content_filtering_features.parquet') #input
        self.book_content_features_path_output = os.path.join(self.model_dir_personalized, 'gr_book_content_features.parquet') #output
        self.scaler_path_output = os.path.join(self.model_dir_personalized, 'gr_content_feature_scaler.pkl') #output
        self.book_primary_authors_path_output = os.path.join(self.model_dir_personalized, 'gr_book_primary_authors.parquet') #output

    def load(self):
        print(f'GRBookContentFeatureBuilder: loading content filtering features from {self.content_features_path_input}')
        self.df = pd.read_parquet(self.content_features_path_input)
        print(f'GRBookContentFeatureBuilder: loaded {len(self.df)} books.')

    def _normalize_genre_weights(self, genres):
        # genres is a dict like {'children': 6.0, 'fiction': None, ...}
        # assign weights proportional to the count associated with the genre 
        if not isinstance(genres, dict):
            return [0.0] * len(GENRE_KEYS)
        counts = [genres.get(key) or 0.0 for key in GENRE_KEYS]
        total = sum(counts)
        if total == 0:
            return [0.0] * len(GENRE_KEYS)
        return [c / total for c in counts]

    def _encode_language(self, lang):
        # map language code to category, one-hot encode
        # return a list that has 1.0 for passed in valid language but zero for others
        if not isinstance(lang, str):
            category = 'other'
        else:
            category = LANGUAGE_MAP.get(lang.strip(), 'other')
        return [1.0 if category == cat else 0.0 for cat in LANGUAGE_CATEGORIES]

    def _encode_primary_authors(self):
        # extract primary authors (role='') per book for hybrid model
        print('GRBookContentFeatureBuilder: extracting primary authors...')
        rows = []
        all_author_ids = set()

        for _, row in self.df.iterrows():
            authors = row['authors']
            if not isinstance(authors, (list, np.ndarray)):
                continue
            # note: there could be multiple primary authors for a given book, hence list!
            primary = [a['author_id'] for a in authors if a.get('role') == '']
            if not primary:
                # pick any and all author_id's if primary authors i.e authors with role='' missing!
                primary = [a['author_id'] for a in authors if a.get('author_id')]
            for author_id in primary:
                all_author_ids.add(author_id)
                # assign one row for each author of the same book
                rows.append({'book_id': row['book_id'], 'book_id_csv': row['book_id_csv'], 'author_id': author_id})

        # build author_id_csv mapping, sorting for consistency on id assignments for authors across runs!
        # we are assigning 0....k numbers for each unique author and naming that id/index as author_id_csv
        # plus one is to avoid padding ambiguity with leading zero
        # if 0 0 0 means, first zero could be no-author case (or) 0th author as the primary author case
        # if we shift all author_id_csv's by one, the there is no 0 user so first 0 is certainly padding case i.e no primary user
        author_id_map = {aid: idx + 1 for idx, aid in enumerate(sorted(all_author_ids))}
        author_df = pd.DataFrame(rows)
        author_df['author_id_csv'] = author_df['author_id'].map(author_id_map)

        # save author_id_map
        author_id_map_df = pd.DataFrame(list(author_id_map.items()), columns=['author_id', 'author_id_csv'])
        author_id_map_df.to_csv(os.path.join(self.data_dir, 'gr_author_id_map.csv'), index=False)
        print(f'GRBookContentFeatureBuilder: found {len(all_author_ids)} unique authors.')
        return author_df

    def build(self):
        # build() transforms raw book metadata into a 20-dimensional numerical feature vector per book
        # ready to be consumed by the hybrid neural network.
        #
        # the 20 features are:
        #   - 10 genre features: normalized weights per genre (e.g. 0.5 history, 0.3 children, 0.2 fiction)
        #   - 6 language features: one-hot encoding (e.g. [1,0,0,0,0,0] for english)
        #   - 4 numerical features: min-max normalized to [0,1] (publication_year, sentiment, rating_mean, review_count)
        #
        # scaler is fit on training books only to avoid data leakage from val and test sets
        # scaler is saved to disk for consistent scaling at inference time
        # out-of-distribution values are clipped to [0, 1]
        #
        # additionally extracts primary authors per book and saves to disk
        # for use as author embeddings in the hybrid model
        print('GRBookContentFeatureBuilder: encoding features...')

        # encode genres
        genre_features = self.df['genres'].apply(self._normalize_genre_weights)
        genre_df = pd.DataFrame(genre_features.tolist(), columns=[f'genre_{k.replace(", ", "_").replace(" ", "_")}' for k in GENRE_KEYS])

        # encode language, one shot representation
        lang_features = self.df['language_code'].apply(self._encode_language)
        lang_df = pd.DataFrame(lang_features.tolist(), columns=[f'lang_{cat}' for cat in LANGUAGE_CATEGORIES])

        # numerical features
        numerical_cols = ['publication_year', 'weighted_compound_sentiment_score_avg', 'rating_mean', 'review_count']
        num_df = self.df[numerical_cols].copy()

        # convert to float — some columns may be stored as strings in parquet
        for col in numerical_cols:
            num_df[col] = pd.to_numeric(num_df[col], errors='coerce')

        # min-max scaling
        # col_min = 1800
        # col_max = 2023
        # book published in 1800: (1800 - 1800) / (2023 - 1800) = 0/223 = 0.0  ← oldest
        # book published in 2023: (2023 - 1800) / (2023 - 1800) = 223/223 = 1.0  ← newest
        # book published in 1950: (1950 - 1800) / (2023 - 1800) = 150/223 = 0.67
        # basically maps smallest value to 0 and largest value to 1
        # and everything else in between 0 and 1

        # fit scaler on training books only to avoid data leakage from val and test sets
        train_df = pd.read_parquet(os.path.join(self.data_dir_personalized, 'gr_train_reviews.parquet'))
        train_book_ids = set(train_df['book_id_csv'].unique())
        train_mask = self.df['book_id_csv'].isin(train_book_ids)

        scaler = {}
        for col in numerical_cols:
            col_min = num_df.loc[train_mask, col].min()  # fit on training books only
            col_max = num_df.loc[train_mask, col].max()  # fit on training books only
            scaler[col] = {'min': col_min, 'max': col_max}
            if col_max > col_min:
                num_df[col] = (num_df[col] - col_min) / (col_max - col_min)
                # clip out-of-distribution values to [0, 1]
                # e.g. new book published in 2025 when col_max=2023 → 1.009 → clipped to 1.0
                # e.g. new book published in 1750 when col_min=1800 → -0.224 → clipped to 0.0
                num_df[col] = num_df[col].clip(0.0, 1.0)
            else:  # min == max case — all values are the same, set to 0
                num_df[col] = 0.0
        num_df = num_df.fillna(0.0)

        # save scaler so that we can use it during inference
        # let's say if a new book came in during inference, and it has published year 1990
        # we should scale that value using the same scale we used during training
        # that way our scaling is consistent
        with open(self.scaler_path_output, 'wb') as f:
            pickle.dump(scaler, f)
        print(f'GRBookContentFeatureBuilder: saved scaler to {self.scaler_path_output}')

        # combine all features
        self.feature_df = pd.concat([
            self.df[['book_id', 'book_id_csv']].reset_index(drop=True),
            genre_df.reset_index(drop=True),
            lang_df.reset_index(drop=True),
            num_df.reset_index(drop=True)
        ], axis=1)
        print(f'GRBookContentFeatureBuilder: built feature matrix with shape {self.feature_df.shape}')

        # extract primary authors for hybrid model
        # hybrid model will average embeddings of all primary authors per book
        self.author_df = self._encode_primary_authors()

    def save(self):
        print('GRBookContentFeatureBuilder: saving outputs...')
        self.feature_df.to_parquet(self.book_content_features_path_output, index=False)
        print(f'GRBookContentFeatureBuilder: saved book_content_features to {self.book_content_features_path_output}')
        self.author_df.to_parquet(self.book_primary_authors_path_output, index=False)
        print(f'GRBookContentFeatureBuilder: saved book_primary_authors to {self.book_primary_authors_path_output}')

    def run(self, override=False):
        print(f'GRBookContentFeatureBuilder: starting with override={override}')
        if not override and os.path.exists(self.book_content_features_path_output) and os.path.exists(self.book_primary_authors_path_output):
            print('GRBookContentFeatureBuilder: outputs already exist, skipping. Use override=True to regenerate.')
            return
        self.load()
        self.build()
        self.save()
        print('GRBookContentFeatureBuilder: done.')
