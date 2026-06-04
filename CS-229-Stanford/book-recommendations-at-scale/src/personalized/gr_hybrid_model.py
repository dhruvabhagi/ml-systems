#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

class HybridReviewsDataset(Dataset):
    def __init__(self, reviews_df, book_content_features_df, book_primary_authors_df, max_authors):
        self.max_authors = max_authors

        # merge content features into reviews
        reviews_df = reviews_df.merge(book_content_features_df, on='book_id_csv', how='left')

        # merge primary authors into reviews
        # group author_id_csv per book into a list
        authors_grouped = book_primary_authors_df.groupby('book_id_csv')['author_id_csv'].apply(list).reset_index()
        authors_grouped.columns = ['book_id_csv', 'author_id_csvs']
        reviews_df = reviews_df.merge(authors_grouped, on='book_id_csv', how='left')

        # fill missing author lists with empty list
        reviews_df['author_id_csvs'] = reviews_df['author_id_csvs'].apply(
            lambda x: x if isinstance(x, list) else []
        )

        # store tensors
        self.users = torch.tensor(reviews_df['user_id_csv'].values, dtype=torch.long)
        self.books = torch.tensor(reviews_df['book_id_csv'].values, dtype=torch.long)
        self.ratings = torch.tensor(reviews_df['rating'].values, dtype=torch.float32)

        # content feature columns
        content_cols = [c for c in reviews_df.columns if c.startswith('genre_') or c.startswith('lang_') or
                       c in ['publication_year', 'weighted_compound_sentiment_score_avg', 'rating_mean', 'review_count']]
        self.content_features = torch.tensor(reviews_df[content_cols].values, dtype=torch.float32)

        # pad author ids to max_authors length with 0 (padding_idx)
        padded_authors = []
        for author_list in reviews_df['author_id_csvs']:
            padded = author_list[:max_authors]  # truncate if somehow > max_authors
            padded = padded + [0] * (max_authors - len(padded))  # pad with 0
            padded_authors.append(padded)
        self.authors = torch.tensor(padded_authors, dtype=torch.long)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.books[idx], self.authors[idx], self.content_features[idx], self.ratings[idx]


# The actual neural network architecture
# Combines pretrained collaborative filtering embeddings (user, book) with
# author embeddings and content features for improved recommendation quality
# inherits from PyTorch's base neural network class
class NeuralHybridModel(nn.Module):
    def __init__(self, num_users, num_books, num_authors, content_feature_dim, embedding_dim=64):
        super(NeuralHybridModel, self).__init__()

        # user and book embeddings — initialized from pretrained collaborative filtering model
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)

        # author embedding — random init, 0 reserved for padding
        self.author_embedding = nn.Embedding(num_authors + 1, embedding_dim, padding_idx=0)
        nn.init.normal_(self.author_embedding.weight, mean=0, std=0.01)

        # fc layers
        # input: user_emb(64) + book_emb(64) + interaction(64) + author_emb(64) + content(20) = 276
        input_dim = embedding_dim * 4 + content_feature_dim
        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(64, 1)
        )

    def forward(self, user_ids, book_ids, author_ids, content_features):
        user_emb = self.user_embedding(user_ids)
        book_emb = self.book_embedding(book_ids)

        # element-wise product captures explicit user-book interaction signal (He et al. 2017)
        interaction = user_emb * book_emb

        # average author embeddings, ignoring padding (index 0)
        # author_ids shape: (batch, max_authors)
        author_emb = self.author_embedding(author_ids)          # (batch, max_authors, 64)
        mask = (author_ids != 0).float().unsqueeze(-1)          # (batch, max_authors, 1)
        author_emb = (author_emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, 64)

        # concatenate all inputs: user(64) + book(64) + interaction(64) + author(64) + content(20) = 276
        x = torch.cat([user_emb, book_emb, interaction, author_emb, content_features], dim=1)
        return self.fc_layers(x).squeeze()


# The orchestrator/pipeline class
# Handles loading data, building model, training loop, evaluation, saving
class GRHybridModel:
    def __init__(self, embedding_dim=64, batch_size=2048, lr=0.001, num_epochs=50):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.scheduler_patience = 3
        self.early_stopping_patience = 7
        self.scheduler_lr_factor_reduction = 0.5
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model') 
        self.data_dir_personalized = os.path.join(self.data_dir, 'personalized')
        self.model_dir_personalized = os.path.join(self.model_dir, 'personalized')
        self.model_path = os.path.join(self.model_dir_personalized, 'gr_hybrid_model.pt')
        self.cf_model_path = os.path.join(self.model_dir_personalized, 'gr_collaborative_filtering_model.pt')

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'GRHybridModel: using device {self.device}')

    def load(self):
        print('GRHybridModel: loading data...')
        self.train_df = pd.read_parquet(os.path.join(self.data_dir_personalized, 'gr_train_reviews.parquet'))
        self.val_df = pd.read_parquet(os.path.join(self.data_dir_personalized, 'gr_validation_reviews.parquet'))
        self.book_content_features_df = pd.read_parquet(os.path.join(self.model_dir_personalized, 'gr_book_content_features.parquet'))
        self.book_primary_authors_df = pd.read_parquet(os.path.join(self.model_dir_personalized, 'gr_book_primary_authors.parquet'))
        # use full id maps to match CF model embedding table sizes
        user_id_map = pd.read_csv(os.path.join(self.data_dir, 'gr_user_id_map.csv'))
        book_id_map = pd.read_csv(os.path.join(self.data_dir, 'gr_book_id_map.csv'))
        self.num_users = len(user_id_map)
        self.num_books = len(book_id_map)

        self.num_authors = int(self.book_primary_authors_df['author_id_csv'].max())  # +1 handled in model (padding)
        self.max_authors = int(self.book_primary_authors_df.groupby('book_id_csv')['author_id_csv'].count().max())
        self.content_feature_dim = len([c for c in self.book_content_features_df.columns
                                       if c.startswith('genre_') or c.startswith('lang_') or
                                       c in ['publication_year', 'weighted_compound_sentiment_score_avg',
                                             'rating_mean', 'review_count']])

        print(f'GRHybridModel: num_users={self.num_users}, num_books={self.num_books}, num_authors={self.num_authors}')
        print(f'GRHybridModel: max_authors={self.max_authors}, content_feature_dim={self.content_feature_dim}')
        print('GRHybridModel: loaded data successfully.')

    def build(self):
        print('GRHybridModel: building model...')
        self.model = NeuralHybridModel(
            self.num_users, self.num_books, self.num_authors,
            self.content_feature_dim, self.embedding_dim
        ).to(self.device)

        # load pretrained user and book embeddings from collaborative filtering model
        cf_checkpoint = torch.load(self.cf_model_path, map_location=self.device)
        self.model.user_embedding.weight = nn.Parameter(cf_checkpoint['user_embedding.weight'])
        self.model.book_embedding.weight = nn.Parameter(cf_checkpoint['book_embedding.weight'])
        print('GRHybridModel: loaded pretrained user and book embeddings from collaborative filtering model.')

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.scheduler_patience,
            factor=self.scheduler_lr_factor_reduction
        )
        print('GRHybridModel: model built successfully.')

    def train(self):
        if os.path.exists(self.model_path):
            print('GRHybridModel: loading existing model from disk...')
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print('GRHybridModel: loaded existing model successfully.')
            return

        print('GRHybridModel: training model...')
        train_dataset = HybridReviewsDataset(self.train_df, self.book_content_features_df, self.book_primary_authors_df, self.max_authors)
        val_dataset = HybridReviewsDataset(self.val_df, self.book_content_features_df, self.book_primary_authors_df, self.max_authors)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        # phase 1: freeze pretrained user and book embeddings
        # train only author embeddings and fc_layers until they stabilize
        # this prevents noisy gradients from new layers disturbing pretrained representations
        print('GRHybridModel: phase 1 — freezing pretrained user and book embeddings...')
        self.model.user_embedding.weight.requires_grad = False
        self.model.book_embedding.weight.requires_grad = False

        phase1_epochs = 5
        for epoch in range(phase1_epochs):
            self.model.train()
            train_loss = 0
            for user_ids, book_ids, author_ids, content_features, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                book_ids = book_ids.to(self.device)
                author_ids = author_ids.to(self.device)
                content_features = content_features.to(self.device)
                ratings = ratings.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(user_ids, book_ids, author_ids, content_features)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for user_ids, book_ids, author_ids, content_features, ratings in val_loader:
                    user_ids = user_ids.to(self.device)
                    book_ids = book_ids.to(self.device)
                    author_ids = author_ids.to(self.device)
                    content_features = content_features.to(self.device)
                    ratings = ratings.to(self.device)
                    predictions = self.model(user_ids, book_ids, author_ids, content_features)
                    val_loss += self.criterion(predictions, ratings).item()

            train_rmse = (train_loss / len(train_loader)) ** 0.5
            val_rmse = (val_loss / len(val_loader)) ** 0.5
            print(f'GRHybridModel: phase1 epoch={epoch+1}, train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f'GRHybridModel: saved best model at phase1 epoch {epoch+1}')

        # phase 2: unfreeze everything and fine-tune jointly at lower lr
        # fc_layers and author embeddings are now stabilized
        # pretrained embeddings can adapt gently without being disturbed
        print('GRHybridModel: phase 2 — unfreezing all parameters for joint fine-tuning...')
        self.model.user_embedding.weight.requires_grad = True
        self.model.book_embedding.weight.requires_grad = True

        # reload best weights from phase 1 before fine-tuning
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))

        # lower lr for fine-tuning to prevent large updates to pretrained embeddings
        phase2_lr = self.lr * 0.1  # 0.0001
        self.optimizer = optim.Adam(self.model.parameters(), lr=phase2_lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=self.scheduler_patience,
            factor=self.scheduler_lr_factor_reduction
        )

        patience_counter = 0
        best_val_loss = float('inf')

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0
            for user_ids, book_ids, author_ids, content_features, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                book_ids = book_ids.to(self.device)
                author_ids = author_ids.to(self.device)
                content_features = content_features.to(self.device)
                ratings = ratings.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(user_ids, book_ids, author_ids, content_features)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for user_ids, book_ids, author_ids, content_features, ratings in val_loader:
                    user_ids = user_ids.to(self.device)
                    book_ids = book_ids.to(self.device)
                    author_ids = author_ids.to(self.device)
                    content_features = content_features.to(self.device)
                    ratings = ratings.to(self.device)
                    predictions = self.model(user_ids, book_ids, author_ids, content_features)
                    val_loss += self.criterion(predictions, ratings).item()

            train_rmse = (train_loss / len(train_loader)) ** 0.5
            val_rmse = (val_loss / len(val_loader)) ** 0.5
            print(f'GRHybridModel: phase2 epoch={epoch+1}, train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}')

            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f'GRHybridModel: saved best model at phase2 epoch {epoch+1}')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f'GRHybridModel: early stopping at phase2 epoch {epoch+1}')
                    break

            # scheduler step
            curr_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < curr_lr:
                print(f'GRHybridModel: lr reduced to {new_lr}, reloading best weights...')
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f'GRHybridModel: successfully loaded best weights after lr reduction to {new_lr}')

        # load best model
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print('GRHybridModel: training complete.')

    def evaluate(self):
        print('GRHybridModel: evaluating on test set...')
        test_df = pd.read_parquet(os.path.join(self.data_dir_personalized, 'gr_test_reviews.parquet'))
        test_dataset = HybridReviewsDataset(test_df, self.book_content_features_df, self.book_primary_authors_df, self.max_authors)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for user_ids, book_ids, author_ids, content_features, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                book_ids = book_ids.to(self.device)
                author_ids = author_ids.to(self.device)
                content_features = content_features.to(self.device)
                ratings = ratings.to(self.device)
                predictions = self.model(user_ids, book_ids, author_ids, content_features)
                test_loss += self.criterion(predictions, ratings).item()
        test_rmse = (test_loss / len(test_loader)) ** 0.5
        print(f'GRHybridModel: test_rmse={test_rmse:.4f}')
        return test_rmse

    def run(self, override=False):
        print(f'GRHybridModel: starting with override={override}')
        if not override and os.path.exists(self.model_path):
            print(f'GRHybridModel: model file {self.model_path} already exists, skipping. Use override=True to regenerate.')
            return
        print(f'GRHybridModel: starting hybrid model pipeline with override={override}')
        self.load()
        self.load()
        self.build()
        self.train()
        self.evaluate()