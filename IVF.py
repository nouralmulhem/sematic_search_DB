from typing import Dict, List, Annotated
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from file import BinaryFile

class IvfDb:
    def __init__(self, file_path = "saved_db.bin", new_db = True) -> None:
        self.file_path = file_path
        # number of clusters
        self.n_clusters = 50
        # binary file handler
        self.bfh = BinaryFile(self.file_path)
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass
    
    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # insert all rows with bfh
        self.bfh.insert_records(rows)
        #Build index
        self._build_index()
        
    def build(self):
        self._build_index()
        
    def rertive_all(self):
        # return all rows
        return self.bfh.read_all()[:,1:]
              
    def retrive(self, query: Annotated[List[float], 70], top_k = 5):
        scores = []
        centroids = BinaryFile('centroids.bin').read_all()[:,1:]
        for i, centroid in enumerate(centroids):
            
            score_centroid = self._cal_score(query, centroid)
            id = i
            print(id)
            scores.append((score_centroid, id))
        scores = sorted(scores, reverse=True)[:3]
        
        top_15 = []
        for score in scores:
          region_vectors = BinaryFile(f'cluster_{score[1]}.bin').read_all()
          vector_scores = []
          for vec in region_vectors:
            id = vec[0]
            embed = vec[1:]
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
      
      # read all rows
      rows = self.bfh.read_all()
      # ids of the record
      ids = rows[:,0]
      # list of the vectors (70 float each)
      embeds = rows[:,1:]
      # set the number of clusters
      n_clusters = self.n_clusters
      kmeans = KMeans(n_clusters=n_clusters)
      kmeans.fit(embeds)
      # get the labels id of each cluster list of size db each vector to its cluster
      cluster_labels = kmeans.labels_ 
      # centroids which are list of vectors (70 float each)
      centroids = kmeans.cluster_centers_.tolist() 
      # assign each vector to its cluster
      clusters = [[] for _ in range(n_clusters)]        
      for i, label in enumerate(cluster_labels):
        clusters[label].append(tuple((ids[i], embeds[i])))
      # # show the all centriods
      # for cluster_index, centroid in enumerate(centroids):
      #   print(f"Centroid of Cluster {cluster_index}: {centroid}")
      # # show clusters after making them
      # for cluster_index, cluster_vectors in enumerate(clusters):
      #   print(f"Cluster {cluster_index} has {len(cluster_vectors)} vectors.")

      # # show cluster 0 as example
      # for index, vector in enumerate(clusters[0]):
      #   print(f"Cluster0 {index} is {vector} vectors.")

      # insert centroids
      centroids_dict = [{"id": i, "embed": row} for i, row in enumerate(centroids)]
      centroids_file_name = 'centroids.bin'
      open(centroids_file_name, 'w').close()
      bfh2 = BinaryFile(centroids_file_name)
      bfh2.insert_records(centroids_dict)
      # insert clusters
      # insert each cluster in a file
      for cluster_index, cluster_vectors in enumerate(clusters):
        cluster_dict = [{"id": int(row[0]), "embed": list(*row[1:])} for row in cluster_vectors]
        cluster_file_name = f'cluster_{cluster_index}.bin'
        open(cluster_file_name, 'w').close()
        bfh3 = BinaryFile(cluster_file_name)
        bfh3.insert_records(cluster_dict)
      # self.clusters = clusters
      # self.centroids = centroids
      


