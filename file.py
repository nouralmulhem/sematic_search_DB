import random
import struct
from typing import Annotated, Dict, List
import numpy as np
import time

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
            position = row_id * (self.int_size + self.vec_size * self.float_size)  # Size of one row (ID + vec_size * floats)
            # Seek to the position of the row
            file.seek(position)
            # Read the row
            packed_data = file.read(self.int_size + self.vec_size * self.float_size)  # Size of one row (ID + vec_size * floats)
            data = struct.unpack(f'i{self.vec_size}f', packed_data)
            return np.array(data)

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        with open(self.filename, 'ab') as file:
            for row in rows:
                id, embed = row["id"], row["embed"]
                # Pack the ID and the float values into a binary format
                packed_data = struct.pack(f'i{self.vec_size}f', id, *embed)
                # Write the packed data to the file
                file.write(packed_data)

    # read all rows
    def read_all(self):
        rows = []
        with open(self.filename, 'rb') as file:
            # iterate over all rows
            while True:
                # Read the row
                packed_data = file.read(self.int_size + self.vec_size * self.float_size)
                if packed_data == b'':
                    break
                data = struct.unpack(f'i{self.vec_size}f', packed_data)
                rows.append(data)
        return np.array(rows)


def test():
    # setup data
    num_rows = 1000000
    num_tests = 100000
    vec_size = 70
    # define instance of class
    file_path = "data.bin"
    # empty file if exists
    open(file_path, 'w').close()
    bfh = BinaryFile(file_path)
    # create data and write to binary file
    records_np = np.random.random((num_rows, vec_size))
    # ##insert row by row
    # for i in range(num_rows):
    #     bfh.insert_row(i,records_np[i])
    records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
    tic = time.time()
    bfh.insert_records(records_dict)
    toc = time.time()
    np_insert_time = toc - tic
    print(f'The time needed to insert {num_rows} is {np_insert_time}')

    count_test = 0
    failed_ids = []
    for i in range(num_tests):
        random_row_id = random.randint(0, (num_tests - 1))
        vec_ran = bfh.read_row(random_row_id)[1:]
        vec_real = records_np[random_row_id]
        # compare the 2 vectors if one is not equal break
        # if np.array_equal(vec_ran,vec_real):
        if np.allclose(vec_ran,vec_real):
            count_test += 1
        else:
            failed_ids.append(random_row_id)

    print(f"Passed {count_test} tests")
    print(f"Failed {len(failed_ids)} tests")
    print(f"Failed ids: {failed_ids}")

    retrieved_all = bfh.read_all()
    # remove id from retrieved all
    retrieved_all = retrieved_all[:,1:]
    # compare it with records_np
    if np.allclose(retrieved_all,records_np):
        print('all retrieved data are equal')
    else:
        print('all retrieved data are not equal')


    # ## uncomment to print the vector
    # print(records_np[1])
    # print('---------------------')
    # print(bfh.read_row(1))
    # if np.allclose(records_np[1],bfh.read_row(1)[1:]):
    #         print('-----------eq----------')

    # else:
    #        print('-----------noteq----------')

        
if __name__ == '__main__':
    test()