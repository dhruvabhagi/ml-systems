import tqdm
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from gr_i_metadata_handler import IMetadataHandler

class ReviewMetadataHandler(IMetadataHandler):
    def __init__(self):
        self.reviews_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'gr_reviews_dedup.json')
        self.user_id_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'gr_user_id_map.csv')

    def _get_parquet_path(self):
        return self.reviews_data_path.replace('.json', '.parquet')
    
    def iter_reviews_for_scoring(self, batch_size=500_000):
        parquet_file = pq.ParquetFile(self._get_parquet_path())
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=['book_id', 'user_id', 'review_id', 'review_text', 'rating', 'n_votes', 'read_at']):
            yield batch.to_pandas()

    def load(self):
        reviews_data_path_parquet = self._get_parquet_path()
        if not os.path.exists(reviews_data_path_parquet):
            print('ReviewMetadataHandler: converting reviews data from JSON to Parquet for faster loading...')
            writer = None
            with tqdm.tqdm() as pbar:
                for chunk in pd.read_json(self.reviews_data_path, lines=True, chunksize=10_000):
                   # handle the mismatching data types by explicitly converting them to strings
                   chunk = chunk.astype({'read_at': str, 'started_at': str})
                   table = pa.Table.from_pandas(chunk)
                   if writer is None:
                       writer = pq.ParquetWriter(reviews_data_path_parquet, table.schema)
                   writer.write_table(table)  # ← writes to disk immediately, no concat!
                   pbar.update(len(chunk))
            writer.close()
            print('ReviewMetadataHandler: successfully converted reviews data from JSON to Parquet.')
        print('ReviewMetadataHandler: reading the reviews data from Parquet file...')

        def do_load(): # intentionally doesn't load review_text!
            parquet_file = pq.ParquetFile(reviews_data_path_parquet)
            chunks = []
            for batch in parquet_file.iter_batches(
                batch_size=500_000,
                columns=['book_id', 'user_id', 'review_id', 'rating', 'n_votes', 'read_at']
            ):
                print("ReviewMetadataHandler: loaded a new batch of reviews metadata from Parquet file...")
                chunks.append(batch.to_pandas())
            self.df = pd.concat(chunks, ignore_index=True)

        do_load()
        print('ReviewMetadataHandler: successfully read reviews metadata from Parquet file.')

    def preprocess(self):
        self.df = self.df.dropna(subset=['book_id']) # dropna drops rows where there is a missing value for any of these fields
        user_id_map = pd.read_csv(self.user_id_map_path)
        self.df = pd.merge(self.df, user_id_map, on='user_id', how='left')
        print(f'ReviewMetadataHandler: Null user_id_csv: {self.df["user_id_csv"].isna().sum()}')
        print('ReviewMetadataHandler: reviews metadata preprocessed successfully.')

    def get(self):
         return self.df
       