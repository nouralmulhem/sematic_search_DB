import struct
from typing import Annotated, Dict, List
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neighbors import LSHForest
from dataclasses import dataclass

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

    def read_record(self, record_index):
        # Open the file in binary mode
        with open(self.filename, "rb") as file:
            # Calculate the size of each record based on the structure of the data
            record_size = struct.calcsize('!I') + (self.vec_size) * struct.calcsize('!f')

            # Seek to the beginning of the specified record
            file.seek(record_index * record_size, 0)

            # Read the record
            record_binary = file.read(record_size)

            if not record_binary:
                return None  # End of file or invalid record index

            # Unpack the binary data into a list of integers and floats
            unpacked_data = struct.unpack('!' + 'I' + 'f' * self.vec_size, record_binary)

        return unpacked_data

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        first_position = None
        last_position = None
        with open(self.filename, 'ab') as file:
            # record the position before writing
            first_position = file.tell()
            for row in rows:
                id, embed = row["id"], row["embed"]
                embed.insert(0,id)
                # Pack the ID and the float values into a binary format
                packed_data = struct.pack('!I' + 'f' * (len(embed) - 1), *embed)
                # Write the packed data to the file
                file.write(packed_data)
            # Record the position after writing
            last_position = file.tell()
        # Return the first and last position
        return first_position, last_position

    # read all rows
    def read_all(self):
        # Calculate the size of each record based on the structure of the data
        record_size = struct.calcsize('!I') + (self.vec_size) * struct.calcsize('!f')
        rows = []
        with open(self.filename, 'rb') as file:
            # iterate over all rows
            while True:
                # Read the row
                packed_data = file.read(record_size)
                if packed_data == b'':
                    break
                data = struct.unpack('!I' + 'f' * self.vec_size, packed_data)
                rows.append(data)
        return np.array(rows, dtype=np.float32)
    
    def read_records(self, start_record, end_record):
        # Open the file in binary mode
        with open(self.filename, "rb") as file:
            # Calculate the size of each record based on the structure of the data
            record_size = struct.calcsize('!I') + (self.vec_size) * struct.calcsize('!f')

            # Seek to the beginning of the start record
            file.seek(start_record * record_size, 0)

            # Read the specified range of records
            records = []
            for _ in range(start_record, end_record):
                record_binary = file.read(record_size)

                if not record_binary:
                    break  # End of file

                # Unpack the binary data into a list of integers and floats
                unpacked_data = struct.unpack('!' + 'I' + 'f' * self.vec_size, record_binary)

                # Append the record to the list
                records.append(unpacked_data)

        return np.array(records, dtype=np.float32)

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
    


# insertion if saved_db is not created

class CreateDatabase:
    def __init__(self, file_path="saved_db.bin", new_db=True) -> None:
        self.file_path = file_path
        # binary file handler
        self.bfh = BinaryFile(self.file_path)
        if new_db:
            # just open new file to delete the old one
            with open(self.file_path, "ab") as fout:
                # if you need to add any head to the file
                pass

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        # insert all rows with bfh
        self.bfh.insert_records(rows)
    
def cal_score(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
    return cosine_similarity

# db = CreateDatabase(file_path="saved_db.bin")
# records_np = np.random.random((1000, 70))
# records_dict = [{"id": i, "embed": list(row)}
#                 for i, row in enumerate(records_np)]
# db.insert_records(records_dict)