import tqdm
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gr_i_metadata_handler import IMetadataHandler

class BookMetadataHandler(IMetadataHandler):
    def __init__(self):
        self.book_metadata_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'gr_books.json')
        self.book_genres_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'gr_book_genres_initial.json')
        self.book_id_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'gr_book_id_map.csv')

    def load(self):
        book_metadata_path_parquet = self.book_metadata_path.replace('.json', '.parquet')
        if not os.path.exists(book_metadata_path_parquet):
            print('BookMetadataHandler: converting book metadata from JSON to Parquet for faster loading...')
            writer = None
            with tqdm.tqdm() as pbar:
                for chunk in pd.read_json(self.book_metadata_path, lines=True, chunksize=10_000):
                    # handle the mismatching data types by explicitly converting them to strings
                    chunk = chunk.astype({
                        'text_reviews_count': str,
                        'average_rating': str,
                        'ratings_count': str,
                        'work_id': str,
                    })
                    table = pa.Table.from_pandas(chunk)
                    if writer is None:
                        writer = pq.ParquetWriter(book_metadata_path_parquet, table.schema)
                    writer.write_table(table)  # ← writes to disk immediately, no concat!
                    pbar.update(len(chunk))
            writer.close()
            print('BookMetadataHandler: successfully converted book metadata from JSON to Parquet.')
        print(f'BookMetadataHandler: reading the book metadata from Parquet file {book_metadata_path_parquet}') 
        self.book_metadata = pd.read_parquet(book_metadata_path_parquet)
        print('BookMetadataHandler: successfully read the metadata read from the Parquet file')

        book_genres_path_parquet = self.book_genres_path.replace('.json', '.parquet')
        if not os.path.exists(book_genres_path_parquet):
            print('BookMetadataHandler: converting book genres from JSON to Parquet for faster loading...')
            writer = None
            with tqdm.tqdm() as pbar:
                for chunk in pd.read_json(self.book_genres_path, lines=True, chunksize=10_000):
                   table = pa.Table.from_pandas(chunk)
                   if writer is None:
                       writer = pq.ParquetWriter(book_genres_path_parquet, table.schema)
                   writer.write_table(table)  # ← writes to disk immediately, no concat!
                   pbar.update(len(chunk))
            writer.close()
            print('BookMetadataHandler: successfully converted book genres from JSON to Parquet.')
        print('BookMetadataHandler: reading the book genres from Parquet file...')
        self.book_genres = pd.read_parquet(book_genres_path_parquet)
        print('BookMetadataHandler: successfully read the book genres from the Parquet file')

    def preprocess(self):
        self.book_metadata = self.book_metadata.dropna(subset=['book_id', 'title']) # dropna drops rows where there is a missing value for any of these fields
        self.book_genres = self.book_genres.dropna(subset=['book_id', 'genres'])
        self.book_metadata = pd.merge(self.book_metadata, self.book_genres, on='book_id', how='left') # for now, let's keep all books, even those witout genres!
        book_id_map = pd.read_csv(self.book_id_map_path)
        # ensure matching types before merge
        self.book_metadata = pd.merge(self.book_metadata, book_id_map, on='book_id', how='left')
        # there are 5 books without a mapping but they aren't part of our reviews so we can safely ignore
        # not part of our reviews because they don't have enough ratings (our minimum of 50 ratings for inclusion in to our ML process)
        print(f'BookMetadataHandler: Null book_id_csv: {self.book_metadata["book_id_csv"].isna().sum()}')
        print('BookMetadataHandler: book metadata preprocessed successfully.')
    
    def get(self):
        return self.book_metadata
