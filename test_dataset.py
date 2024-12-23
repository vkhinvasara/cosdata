import time
import numpy as np
import polars as pl
from random import sample
from scipy.spatial.distance import cdist
from test import abort_transaction, commit_transaction, login, create_db, create_transaction, upsert_in_transaction, ann_vector
from concurrent.futures import ThreadPoolExecutor
import random


DATASET_URL = 'hf://datasets/ashraq/cohere-wiki-embedding-100k/data/train-*-of-*.parquet'
constants = {
    "COLLECTION_NAME":"TestDataset",
    "DIMENSIONS":None,
    "MAX_VAL":1.0,
    "MIN_VAL": -1.0,
    "BATCH_SIZE": 500,
    "BATCH_COUNT": None
}
parquet_path = 'dataset.parquet'
csv_path = 'dataset_brute_force_results.csv'
def setup_dataset():
    try:
        print("Attempting to read parquet file.")
        df = pl.read_parquet(parquet_path)
        print("Loaded the dataset successfully")
    except FileNotFoundError:
        print("Parquet file not found, downloading......")
        df = pl.read_parquet(DATASET_URL)
        df = df[["id", "emb"]]
        df = df.with_columns([
            pl.col("emb").map_elements(lambda emb: [max(-0.9878, min(float(v), 0.987890)) for v in emb], return_dtype=pl.List(pl.Float64))
        ])
        df.write_parquet(parquet_path)
    return df

def generate_brute_force_results():
    try:
        print("Attempting to read pre-generated brute force results.")
        loaded_df = pl.read_csv(csv_path)
        parquet_df = pl.read_parquet(parquet_path)
        print(f"Successfully loaded results from {csv_path}")
        constants['DIMENSIONS'] = max(len(emb) for emb in parquet_df['emb'])
    except FileNotFoundError:
        print(f"{csv_path} not found, generating new brute force results...")
        dataset = setup_dataset()
        parquet_df = dataset
        dataset = dataset.rows()
        vecs = np.array([row[1] for row in dataset], dtype=float) 
        ids = np.array([int(row[0]) for row in dataset], dtype=int)
        total_vectors = len(vecs)
        constants['DIMENSIONS'] = max(len(emb) for emb in parquet_df['emb'])
        print(f"Total vectors read from parquet files: {total_vectors}")
        np.random.seed(42)
        query_indices = np.random.choice(total_vectors, 100, replace=False)
        query_vecs = vecs[query_indices]
        query_ids = ids[query_indices]

        results = []
        for i, qv in enumerate(query_vecs):
            dists = cdist(qv[None, :], vecs, metric='cosine').flatten() 
            sims = 1 - dists  

            top5_idx = np.argpartition(sims, -6)[-6:]
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
                'top5_id': top5[4][0],
                'top5_sim': top5[4][1]
            })
            print(f"Completed generating similarities for {i+1} queries.")

        results_df = pl.DataFrame(results)
        results_df.write_csv(csv_path)
        print("Finished calculating brute-force similarities")

def upsert_dataset_vectors():
    df = pl.read_parquet(parquet_path)
    vectors = []
    for row in df.iter_rows():
        corrected_values = [max(-0.9878, min(float(v), 0.987890)) for v in row[1]]
        vector = {
            "id": int(row[0]),
            "values": corrected_values  
        }
        vectors.append(vector)

    constants['TOKEN'] = login()
    create_db_response = create_db(constants["COLLECTION_NAME"], "Embeddings from dataset", constants['DIMENSIONS'])

    def upsert_with_retry(start_idx, retries=3):
        for attempt in range(retries):
            try:
                upsert_in_transaction(constants["COLLECTION_NAME"], txn_id, vectors[start_idx : start_idx + batch_size])
                return
            except Exception as e:
                print(f"Upsert attempt {attempt + 1} failed for batch starting at index {start_idx}: {e}")
                time.sleep(random.uniform(1, 3)) 
        print(f"Failed to upsert batch starting at index {start_idx} after {retries} attempts")

    transaction = create_transaction(constants['COLLECTION_NAME'])
    txn_id = transaction["transaction_id"]

    batch_size = constants['BATCH_SIZE']
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(upsert_with_retry, i) for i in range(0, len(vectors), batch_size)]
        for future in futures:
            future.result()

    commit_transaction(constants["COLLECTION_NAME"], txn_id)
    print("Vectors upserted successfully!")

def search_query_vectors():
    query_vecs = pl.read_csv(csv_path)
    df = pl.read_parquet(parquet_path)
    query_embeddings = []

    results = []
    for row in query_vecs.iter_rows():
        query_id = row[0]
        embedding = df.filter(pl.col("id") == query_id)["emb"].to_list()[0]
        if embedding:
            embedding = [max(-0.9878, min(float(v), 0.987890)) for v in embedding]
            search_results = search_dataset_vectors(query_id, embedding)
            top5_results = search_results[1]['RespVectorKNN']['knn'][:5]
            results.append({
                'query_id': query_id,
                'top1_id': top5_results[0][0],
                'top1_sim': top5_results[0][1]['CosineSimilarity'],
                'top2_id': top5_results[1][0],
                'top2_sim': top5_results[1][1]['CosineSimilarity'],
                'top3_id': top5_results[2][0],
                'top3_sim': top5_results[2][1]['CosineSimilarity'],
                'top4_id': top5_results[3][0],
                'top4_sim': top5_results[3][1]['CosineSimilarity'],
                'top5_id': top5_results[4][0],
                'top5_sim': top5_results[4][1]['CosineSimilarity']
            })

    results_df = pl.DataFrame(results)
    results_df.write_csv("dataset_server_results.csv")
    print("Finished storing search results in dataset_server_results.csv")

def search_dataset_vectors(query_id, query_emb):
    result = ann_vector(query_id, constants["COLLECTION_NAME"], query_emb)
    return result 

def compare_csv_results(brute_force_csv="dataset_brute_force_results.csv", server_csv="dataset_server_results.csv"):
    bf_df = pl.read_csv(brute_force_csv)
    server_df = pl.read_csv(server_csv)
    bf_map = {}
    for row in bf_df.iter_rows(named=True):
        bf_map[row["query_id"]] = {
            "top_ids": [row[f"top{i}_id"] for i in range(1, 6)]
        }

    server_map = {}
    for row in server_df.iter_rows(named=True):
        server_map[row["query_id"]] = {
            "top_ids": [row[f"top{i}_id"] for i in range(1, 6)]
        }

    total_recall = 0
    count = 0
    for query_id, bf_obj in bf_map.items():
        if query_id in server_map:
            bf_list = bf_obj["top_ids"]
            if query_id in bf_list:
                bf_list.remove(query_id)
            bf_list = bf_list[:4]
    
            server_list = server_map[query_id]["top_ids"]
            if query_id in server_list:
                server_list.remove(query_id)
            server_list = server_list[:4]
    
            bf_ids = set(bf_list)
            server_ids = set(server_list)
    
            intersection_count = len(bf_ids & server_ids)
            recall = (intersection_count / 4.0) * 100.0
            total_recall += recall
            count += 1
    
    avg_recall = (total_recall / count) if count > 0 else 0
    print(f"Average Recall@4: {avg_recall:.2f}%")

if __name__ == "__main__":
    generate_brute_force_results()
    upsert_dataset_vectors()
    search_query_vectors()
    compare_csv_results()