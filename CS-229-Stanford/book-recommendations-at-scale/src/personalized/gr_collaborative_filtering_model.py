#!/usr/bin/env python3

import os
import sys
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
import pandas as pd
from torch.utils.data import Dataset, DataLoader #type: ignore

# add src to path so we can import from parent directory
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

class ReviewsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['user_id_csv'].values, dtype=torch.long)
        self.books = torch.tensor(df['book_id_csv'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.books[idx], self.ratings[idx]


# The actual neural network architecture
# Defines embedding layers, dense layers, forward pass
# Pure PyTorch — only knows about tensors and gradients
# Think of it as the "brain" of the model that learns to predict ratings based on user and book embeddings
#  inherits from PyTorch's base neural network class so we get
# .to(device), .train(),.eval(), state_dict(), forward() etc methods for free from the base class
class NeuralCollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_books, embedding_dim=64):
        super(NeuralCollaborativeFilteringModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        # explicit embedding initialization with small std to prevent large initial predictions
        nn.init.normal_(self.user_embedding.weight, mean=0, std=0.01)
        nn.init.normal_(self.book_embedding.weight, mean=0, std=0.01)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), # dense layer as every i/p neuron is connected with output neuron
            nn.ReLU(), # activation function
            nn.Dropout(0.35), # regularization layer to avoid overfitting, works by randomly zeroing neurons in each training iteration
            nn.Linear(128, 64), # dense layer as every i/p neuron is connected with output neuron
            nn.ReLU(), # activation function
            nn.Dropout(0.35), # regularization layer to avoid overfitting, works by randomly zeroing neurons in each training iteration
            nn.Linear(64, 1) # dense layer as every i/p neuron is connected with output neuron
        )

    def forward(self, user_ids, book_ids):
        user_emb = self.user_embedding(user_ids)
        book_emb = self.book_embedding(book_ids)
        x = torch.cat([user_emb, book_emb], dim=1)
        return self.fc_layers(x).squeeze()

# The orchestrator/pipeline class
# Handles loading data, building the model, training loop, evaluation, saving
# Knows about parquet files, pandas dataframes, file paths
# Think of it as the "manager" that oversees the entire process of training and evaluating the collaborative filtering model
class GRCollaborativeFilteringModel:
    def __init__(self, embedding_dim=64, batch_size=2048, lr=0.001, num_epochs=50):
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.scheduler_patience = 3
        self.early_stopping_patience = 7
        self.scheduler_lr_factor_reduction = 0.5
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
        self.data_dir_personalized = os.path.join(self.data_dir, 'personalized')
        self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'model')
        self.model_dir_personalized = os.path.join(self.model_dir, 'personalized')
        self.model_path = os.path.join(self.model_dir_personalized, 'gr_collaborative_filtering_model.pt')

        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available(): 
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print(f'GRCollaborativeFilteringModel: using device {self.device}')

    def load(self):
        print('GRCollaborativeFilteringModel: loading train and validation data...')
        self.train_df = pd.read_parquet(os.path.join(self.data_dir_personalized, 'gr_train_reviews.parquet'))
        self.val_df = pd.read_parquet(os.path.join(self.data_dir_personalized, 'gr_validation_reviews.parquet'))
        self.num_users = int(self.train_df['user_id_csv'].max()) + 1
        self.num_books = int(self.train_df['book_id_csv'].max()) + 1
        print(f'GRCollaborativeFilteringModel: num_users={self.num_users}, num_books={self.num_books}')
        print('GRCollaborativeFilteringModel: loaded train and validation data successfully.')

    def build(self):
        print('GRCollaborativeFilteringModel: building model...')
        self.model = NeuralCollaborativeFilteringModel(self.num_users, self.num_books, self.embedding_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', # reduce when val loss stops decreasing
            patience=self.scheduler_patience, # wait scheduler_patience(3) epochs before reducing
            factor=self.scheduler_lr_factor_reduction # multiply lr by scheduler_lr_factor_reduction(0.5)
        )
        print('GRCollaborativeFilteringModel: model built successfully.')

    def train(self):
        if os.path.exists(self.model_path):
            print('GRCollaborativeFilteringModel: loading existing model from disk...')
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            print('GRCollaborativeFilteringModel: loaded existing model successfully.')
            return

        print('GRCollaborativeFilteringModel: training model...')
        train_dataset = ReviewsDataset(self.train_df)
        val_dataset = ReviewsDataset(self.val_df)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf') # positive infinity
        patience_counter = 0

        for epoch in range(self.num_epochs):
            # training
            self.model.train()
            train_loss = 0
            for user_ids, book_ids, ratings in train_loader:
                user_ids = user_ids.to(self.device)
                book_ids = book_ids.to(self.device)
                ratings = ratings.to(self.device)
                self.optimizer.zero_grad()
                predictions = self.model(user_ids, book_ids)
                loss = self.criterion(predictions, ratings)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for user_ids, book_ids, ratings in val_loader:
                    user_ids = user_ids.to(self.device)
                    book_ids = book_ids.to(self.device)
                    ratings = ratings.to(self.device)
                    predictions = self.model(user_ids, book_ids)
                    val_loss += self.criterion(predictions, ratings).item()

            train_rmse = (train_loss / len(train_loader)) ** 0.5
            val_rmse = (val_loss / len(val_loader)) ** 0.5
            print(f'GRCollaborativeFilteringModel: epoch={epoch+1}, train_rmse={train_rmse:.4f}, val_rmse={val_rmse:.4f}')
            
            # early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_path)
                print(f'GRCollaborativeFilteringModel: saved best model at epoch {epoch+1}')
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f'GRCollaborativeFilteringModel: early stopping at epoch {epoch+1}')
                    break
            
            # scheduler step
            curr_lr = self.optimizer.param_groups[0]['lr']  # ← get lr before step
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr < curr_lr:
                # this means scheduler reduced the lr as the val_loss hasn't improved for the last 2 consecutive epochs
                # then go and retrieve the best model we have saved amongst all our epochs and then start from there
                # by applying lower learning rate so that we don't cross over the optimal (minima in our case)
                print(f'GRCollaborativeFilteringModel: lr reduced to {new_lr}, reloading best weights...')
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print(f'GRCollaborativeFilteringModel: successfully loaded the best model weights after reducing the lr to {new_lr}')

        # load best model
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        print('GRCollaborativeFilteringModel: training complete.')

    def evaluate(self):
        print('GRCollaborativeFilteringModel: evaluating on test set...')
        test_df = pd.read_parquet(os.path.join(self.data_dir_personalized, 'gr_test_reviews.parquet'))
        test_dataset = ReviewsDataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False) # avoid loading all at once
        self.model.eval()
        test_loss = 0
        with torch.no_grad(): 
            # why no_grad() -- During training, PyTorch tracks every operation on tensors to compute gradients for backpropagation
            # This tracking uses extra memory and computation. *During evaluation we don't need gradients
            # So `torch.no_grad()` tells PyTorch to stop tracking operations, don't build gradient history
            for user_ids, book_ids, ratings in test_loader:
                user_ids = user_ids.to(self.device)
                book_ids = book_ids.to(self.device)
                ratings = ratings.to(self.device)
                predictions = self.model(user_ids, book_ids) # predicted ratings
                test_loss += self.criterion(predictions, ratings).item()
        test_rmse = (test_loss / len(test_loader)) ** 0.5 # double-start is pow(x, 2)
        print(f'GRCollaborativeFilteringModel: test_rmse={test_rmse:.4f}')
        return test_rmse

    def get_embeddings(self):
        user_embeddings = self.model.user_embedding.weight.detach().cpu().numpy()
        book_embeddings = self.model.book_embedding.weight.detach().cpu().numpy()
        return user_embeddings, book_embeddings

    def run(self, override=False):
        print(f'GRCollaborativeFilteringModel: starting with override={override}')
        if not override and os.path.exists(self.model_path):
            print(f'GRCollaborativeFilteringModel: model file {self.model_path} already exists, skipping. Use override=True to regenerate.')
            return
        print(f'GRCollaborativeFilteringModel: starting CF model pipeline with override={override}')
        self.load()
        self.build()
        self.train()
        self.evaluate()