import struct
from typing import Annotated, Dict, List
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import random
from sklearn.utils import shuffle
import pickle

AVG_OVERX_ROWS = 10


class BinaryFile:
    def __init__(self, filename):
        self.filename = filename
        self.vec_size = 70
        self.float_size = 4
        self.int_size = 4

    def insert_row(self, row_id, row_data):
        with open(self.filename, 'ab') as file:
            # Pack the ID and the float values into a binary format
            packed_data = struct.pack(f'i{self.vec_size}f', row_id, *row_data)
            # Write the packed data to the file
            file.write(packed_data)

    def read_row(self, row_id):
        with open(self.filename, 'rb') as file:
            # Calculate the position of the row
            # Size of one row (ID + vec_size * floats)
            position = row_id * \
                (self.int_size + self.vec_size * self.float_size)
            # Seek to the position of the row
            file.seek(position)
            # Read the row
            # Size of one row (ID + vec_size * floats)
            packed_data = file.read(
                self.int_size + self.vec_size * self.float_size)
            data = struct.unpack(f'i{self.vec_size}f', packed_data)
            return np.array(data)

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        first_position = None
        last_position = None
        with open(self.filename, 'ab') as file:
            # record the position before writing
            first_position = file.tell()
            for row in rows:
                id, embed = row["id"], row["embed"]
                # Pack the ID and the float values into a binary format
                packed_data = struct.pack(f'i{self.vec_size}f', id, *embed)
                # Write the packed data to the file
                file.write(packed_data)
            # Record the position after writing
            last_position = file.tell()
        # Return the first and last position
        return first_position, last_position

    # read all rows
    def read_all(self):
        rows = []
        with open(self.filename, 'rb') as file:
            # iterate over all rows
            while True:
                # Read the row
                packed_data = file.read(
                    self.int_size + self.vec_size * self.float_size)
                if packed_data == b'':
                    break
                data = struct.unpack(f'i{self.vec_size}f', packed_data)
                rows.append(data)
        return np.array(rows)

    def read_positions_in_range(self, first_position, last_position):
        records = []
        with open(self.filename, 'rb') as file:
            file.seek(first_position)
            while file.tell() < last_position:
                packed_data = file.read(
                    self.int_size + self.vec_size * self.float_size)
                if packed_data == b'':
                    break
                data = struct.unpack(f'i{self.vec_size}f', packed_data)
                records.append(data)
        return np.array(records)

    def insert_position(self, row_id, position):
        with open(self.filename, 'ab') as file:
            packed_data = struct.pack('iii', row_id, *position)
            file.write(packed_data)

    def read_position(self, row_id):
        with open(self.filename, 'rb') as file:
            position = row_id * (self.int_size * 2 + self.int_size)
            file.seek(position)
            packed_data = file.read(self.int_size * 3)
            data = struct.unpack('iii', packed_data)
            return np.array(data)

    def insert_positions(self, rows: List[Dict[int, List[int]]]):
        with open(self.filename, 'ab') as file:
            for row in rows:
                id, position = row["id"], row["position"]
                packed_data = struct.pack('iii', id, *position)
                file.write(packed_data)

    def read_all_positions(self):
        positions = []
        with open(self.filename, 'rb') as file:
            while True:
                packed_data = file.read(self.int_size * 3)
                if packed_data == b'':
                    break
                data = struct.unpack('iii', packed_data)
                positions.append(data)
        return np.array(positions)

    # cluster_id id
    def insert_cluster(self, row_id, row_data):
      with open(self.filename, 'ab') as file:
            packed_data = struct.pack('ii', row_id, row_data)
            file.write(packed_data)

    def insert_clusters(self, row_id, rows):
        first_position = None
        last_position = None
        with open(self.filename, 'ab') as file:
            # record the position before writing
            first_position = file.tell()
            for row in rows:
                # Pack the ID and the float values into a binary format
                packed_data = struct.pack('ii', row_id, row)
                # Write the packed data to the file
                file.write(packed_data)
            # Record the position after writing
            last_position = file.tell()
        # Return the first and last position
        return first_position, last_position

    def read_clusters_in_range(self, first_position, last_position):
        records = []
        with open(self.filename, 'rb') as file:
            file.seek(first_position)
            while file.tell() < last_position:
                packed_data = file.read(
                    self.int_size + self.int_size)
                if packed_data == b'':
                    break
                data = struct.unpack('ii', packed_data)
                records.append(data)
        return np.array(records)

    # read all rows
    def read_all_clusters(self):
        rows = []
        with open(self.filename, 'rb') as file:
            # iterate over all rows
            while True:
                # Read the row
                packed_data = file.read(
                    self.int_size + self.int_size)
                if packed_data == b'':
                    break
                data = struct.unpack('ii', packed_data)
                rows.append(data)
        return np.array(rows)




class VecDB:
    def __init__(self, file_path="10k", new_db = True) -> None:
        self.file_path = "/content/saved_db_" + file_path + ".bin"
        self.centroids = "/content/centroids_" + file_path + ".bin"
        self.position = "/content/positions_" + file_path + ".bin"
        self.cluster_path = "/content/cluster_" + file_path + ".bin"
        # number of clusters
        self.n_clusters1 = 100
        # self.n_clusters2 = 10
        # binary file handler
        self.bfh = BinaryFile(self.file_path)
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "w") as fout:
                # if you need to add any head to the file
                pass

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        self.bfh.insert_records(rows)
        self.build_index()

    def rertive_embeddings(self):
        # return all rows
        return self.bfh.read_all()[:, 1:]

    #############################################################
    ############     search with cos similarity     #############
    #############################################################
    def _search_with_cos_similarity(self, position_file, cluster_file, scores_id_array, query, top_in_region_num, top_results_num):
      bfh_c_pos = BinaryFile(position_file)
      bfh_c = BinaryFile(cluster_file)
      top_results = []
      for score in scores_id_array:
          # read position of this cluster index (centroid index)
          first_position, second_position = bfh_c_pos.read_position(int(score[1]))[1:]
          # read all vectors in this cluster as [[], [], [], ...]
          vec_ids = bfh_c.read_clusters_in_range(first_position, second_position)
          region_vectors_scores = []
          for vec_id in vec_ids:
              vec = self.bfh.read_row(int(vec_id[1]))
              # read id and features of this vector
              id = vec[0]
              embed = vec[1:]
              vector_score = self._cal_score(query, embed)
              region_vectors_scores.append((vector_score, id))

          # get k (top_in_region_num) the nearest vectors of that region
          region_vectors_scores = sorted(region_vectors_scores, reverse=True)[:top_in_region_num]
          # concat to get all results of all regions
          top_results = top_results + region_vectors_scores

      # take the best k (top_results_num) vectors in those vectors
      top_results = sorted(top_results, reverse=True)[:top_results_num]

      # the top_results here has scores and ids sorted on scores
      return top_results

    #############################################################
    #############     our rock star retrive     #################
    #############################################################
    def retrive(self, query: Annotated[List[float], 70], top_k=5):
        scores = []
        centroids_level2 = BinaryFile(self.centroids).read_all()
        for centroid in centroids_level2:
            score_centroid = self._cal_score(query, centroid[1:])
            id = centroid[0]
            scores.append((score_centroid, id))
        scores = sorted(scores, reverse=True)[:30]

        top_results_level_1 = self._search_with_cos_similarity(self.position, self.cluster_path, scores, query, 30, top_k)

        # top_results_level_1 = self._search_with_knn('positions_cluster_1.bin', 'cluster_1.bin', scores, query, top_k)

        # here we assume that if two rows have the same score, return the lowest ID
        return [score[1] for score in top_results_level_1]


    #############################################################
    ####################     clc score     ######################
    #############################################################
    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity



    #############################################################
    ###########     save clusters and positions     #############
    #############################################################
    def write_to_file(self, cluster_file_name, position_file_name ,centroids_file_name, clusters, centroids):
        # save all cluster in a file
        # output 2 file cluster, position
        # insert clusters
        # insert each cluster in a file
        open(cluster_file_name, 'w').close()
        open(position_file_name, 'w').close()
        open(centroids_file_name, 'w').close()

        bfh_c = BinaryFile(cluster_file_name)
        bfh_c_pos = BinaryFile(position_file_name)
        bfh_cen = BinaryFile(centroids_file_name)

        for cluster_index, cluster_vectors in enumerate(clusters):
            first_position, last_position = bfh_c.insert_clusters(cluster_index, cluster_vectors)
            bfh_c_pos.insert_position(cluster_index, [first_position, last_position])
        #############################################################

        # insert centroids
        centroids_dict = [{"id": i, "embed": row} for i, row in enumerate(centroids)]
        bfh_cen.insert_records(centroids_dict)

        for cluster_index, cluster_vectors in enumerate(clusters):
          print(f"Cluster {cluster_index} has {len(cluster_vectors)} vectors.")


    #############################################################
    #############      partial train kmeans    ##################
    #############################################################
    def partial_predict(self, rows, n_clusters):

        # training_set = embeds[np.random.randint(len(embeds), size=100000 if len(embeds) > 100000 else 1000)]
        # training_set = shuffle(rows)[:100000, 1:]


        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit([tuple(embed) for embed in rows[:, 1:]])
        # del training_set
        # predict rest of data
        cluster_labels = []
        for embed in rows:
            cluster_id = kmeans.predict([tuple(embed[1:])])
            cluster_labels.append(cluster_id)

        # centroids which are list of vectors (70 float each)
        centroids = kmeans.cluster_centers_.tolist()
        del kmeans
        return centroids, cluster_labels


    #############################################################
    ########     second rock star building the index     ########
    #############################################################
    def build_index(self):


        # read 100 000
        # num_ids = 10000
        # leno = 10 * 1000000
        # ranges = random.sample(range(leno), num_ids)
        # read all rows
        rows = []
        for id in range(10000):
          rows.append(self.bfh.read_row(id))

        rows = np.array(rows)
        print(len(rows))

        ###################### level 1 ######################

        # centroids, cluster_labels = self.kmeans_training(rows[:, 1:], self.n_clusters1)
        # centroids, cluster_labels = self.partial_predict(rows, self.n_clusters1)

        kmeans = KMeans(n_clusters=self.n_clusters1)
        kmeans.fit([tuple(embed) for embed in rows[:, 1:]])
        ###############################################################################
        clusters = [[] for _ in range(self.n_clusters1)]

        veccs = [self.bfh.read_row(id) for id in range(10000)]
        predictions = kmeans.predict([tuple(row[1:]) for row in veccs])
        for id, cluster_id in enumerate(predictions):
            clusters[int(cluster_id)].append(int(veccs[id][0]))

        # centroids which are list of vectors (70 float each)
        centroids = kmeans.cluster_centers_.tolist()



        self.write_to_file(self.cluster_path, self.position, self.centroids, clusters, centroids)
