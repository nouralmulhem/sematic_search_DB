from typing import Dict, List, Annotated
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

class IvfDb:
    def __init__(self, file_path = "saved_db.csv", new_db = True) -> None:
        self.file_path = file_path
        # if new_db:
            # just open new file to delete the old one
            # with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                # pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        with open(self.file_path, "a+") as fout:
            for row in rows:
                id, embed = row["id"], row["embed"]
                row_str = f"{id}," + ",".join([str(e) for e in embed])
                fout.write(f"{row_str}\n")
        self._build_index()
        
    def build(self):
        self._build_index()
        
    def rertive_all(self):
        all_rows = []
        with open(self.file_path, "r") as fin:
          for row in fin.readlines():
              row_splits = row.split(",")
              embed = [float(e) for e in row_splits[1:]]
              all_rows.append(embed)
        return np.array(all_rows)
              
    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        scores = []
        for i, centroid in enumerate(self.centroids):
            score_centroid = self._cal_score(query, centroid)
            id = i
            scores.append((score_centroid, id))
        scores = sorted(scores, reverse=True)[:3]
        
        top_15 = []
        for score in scores:
          region_vectors = self.clusters[score[1]]
          vector_scores = []
          for vec in region_vectors:
            id = vec[0]
            embed = vec[1]
            s = self._cal_score(query, list(embed))
            vector_scores.append((s, id))
          vector_scores = sorted(vector_scores, reverse=True)[:top_k]
          top_15 = top_15 + vector_scores
          
        top_15 = sorted(top_15, reverse=True)[:top_k]
        # here we assume that if two rows have the same score, return the lowest ID
        return [s[1] for s in top_15]
    
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self):
      ids = []
      embeds = []
      
      with open(self.file_path, "r") as fin:
        for row in fin.readlines():
            row_splits = row.split(",")
            id = int(row_splits[0])
            embed = tuple(float(e) for e in row_splits[1:])
            
            ids.append(id)
            embeds.append(embed)
            
        # inertias = []

        # for i in range(1,51):
        #     kmeans = KMeans(n_clusters=i)
        #     kmeans.fit(embeds)
        #     inertias.append(kmeans.inertia_)
        

        n_clusters = 50
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(embeds)
        
        cluster_labels = kmeans.labels_ 
        
        centroids = kmeans.cluster_centers_.tolist() 

        clusters = [[] for _ in range(n_clusters)]        

        for i, label in enumerate(cluster_labels):
          clusters[label].append(tuple((ids[i], embeds[i])))
          
        for cluster_index, centroid in enumerate(centroids):
          print(f"Centroid of Cluster {cluster_index}: {centroid}")

        for cluster_index, cluster_vectors in enumerate(clusters):
          print(f"Cluster {cluster_index} has {len(cluster_vectors)} vectors.")
                    
        self.centroids = centroids
        self.clusters = clusters


