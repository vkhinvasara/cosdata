import time
import numpy as np
import polars as pl
from random import sample
from scipy.spatial.distance import cdist
from test import abort_transaction, commit_transaction, login, create_db, create_transaction, create_vector_in_transaction, ann_vector


DATASET_URL = 'hf://datasets/ashraq/cohere-wiki-embedding-100k/data/train-*-of-*.parquet'
constants = {
    "COLLECTION_NAME":"TestDB",
    "DIMENSIONS":1024,
    "MAX_VAL":1.0,
    "MIN_VAL": -1.0,
    "BATCH_SIZE": None,
    "BATCH_COUNT": 100
}
parquet_path = 'dataset.parquet'
def setup_dataset():
    try:
        print("Attempting to read parquet file.")
        df = pl.read_parquet(parquet_path)
        print("Loaded the dataset successfully")
    except FileNotFoundError:
        print("Parquet file not found, downloading......")
        df = pl.read_parquet(DATASET_URL)
        df = df[["id", "emb"]]
        df.filter(
            pl.col("emb").map_elements(
                lambda x: not any(val < -0.9878 or val > 0.98789 for val in x),
                return_dtype=pl.Float64
            )
        )
        df.write_parquet(parquet_path)
    constants['DIMENSIONS'] = max(len(emb) for emb in df['emb'])
    return df
        

csv_path = 'dataset_brute_force_results.csv'
def generate_brute_force_results():
    try:
        print("Attempting to read pre-generated brute force results.")
        loaded_df = pl.read_csv(csv_path)
        parquet_df = pl.read_parquet(parquet_path)
        print(f"Successfully loaded results from {csv_path}")
    except FileNotFoundError:
        print(f"{csv_path} not found, generating new brute force results...")
        dataset = setup_dataset()
        dataset = dataset.rows()
        vecs = np.array([row[1] for row in dataset], dtype=float) 
        ids = np.array([int(row[0]) for row in dataset], dtype=int)
        total_vectors = len(vecs)
        print(f"Total vectors read from parquet files: {total_vectors}")
        np.random.seed(42)
        query_indices = np.random.choice(total_vectors, 100, replace=False)
        query_vecs = vecs[query_indices]
        query_ids = ids[query_indices]

        results = []
        for i, qv in enumerate(query_vecs):
            dists = cdist(qv[None, :], vecs, metric='cosine').flatten() 
            sims = 1 - dists  

            top5_idx = np.argpartition(sims, -5)[-5:]
            top5_idx = top5_idx[np.argsort(sims[top5_idx])[::-1]]

            top5 = [(ids[idx], sims[idx]) for idx in top5_idx if sims[idx] < 0.98789]
            results.append({
                'query_id': query_ids[i],
                'top1_id': top5[0][0],
                'top1_sim': top5[0][1],
                'top2_id': top5[1][0],
                'top2_sim': top5[1][1],
                'top3_id': top5[2][0],
                'top3_sim': top5[2][1],
                'top4_id': top5[3][0],
                'top4_sim': top5[3][1],
            })
            print(f"Completed generating similarities for {i+1} queries.")

        results_df = pl.DataFrame(results)
        results_df.write_csv(csv_path)
        print("Finished calculating brute-force similarities")
    
    
def get_emb_values(query_id):
    df = pl.read_parquet(parquet_path)
    emb = df.filter(pl.col("id") == query_id).select("emb").to_numpy()
    if emb.size > 0:
        return emb[0]
    else:
        raise ValueError(f"Embedding for query_id {query_id} not found.")
    

def upsert_dataset_vectors():
    df = pl.read_parquet(parquet_path)
    vectors = []
    for row in df.iter_rows():
        vector = {
            "id": int(row[0]),
            "values": row[1]  # implement get_emb_values
        }
        vectors.append(vector)

    constants['TOKEN'] = login()
    create_db_response = create_db(constants["COLLECTION_NAME"], "Embeddings from dataset", constants['DIMENSIONS'])

    transaction = create_transaction(constants['COLLECTION_NAME'])
    txn_id = transaction["transaction_id"]

    # Batch upsert
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        create_vector_in_transaction("dataset_collection", txn_id, vectors[i : i + batch_size])

    commit_transaction("dataset_collection", txn_id)

def search_dataset_vectors(query_id, query_emb):
    # Use ann_vector to search
    result = ann_vector(query_id, "dataset_collection", query_emb)
    print(result)
    
if __name__ == "__main__":
    generate_brute_force_results()
    upsert_dataset_vectors()
    search_dataset_vectors()