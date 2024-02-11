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

    # def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
    #     with open(self.filename, 'ab') as file:
    #         for row in rows:
    #             id, embed = row["id"], row["embed"]
    #             # Pack the ID and the float values into a binary format
    #             packed_data = struct.pack(f'i{self.vec_size}f', id, *embed)
    #             # Write the packed data to the file
    #             file.write(packed_data)

    def insert_records(self, rows: List[Dict[int, Annotated[List[float], 70]]]):
        first_position = None
        last_position = None
        with open(self.filename, 'ab') as file:
            #record the position before writing
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
                packed_data = file.read(self.int_size + self.vec_size * self.float_size)
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
                packed_data = file.read(self.int_size + self.vec_size * self.float_size)
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
    _len = len(records_np) 
    # ##insert row by row
    # for i in range(num_rows):
    #     bfh.insert_row(i,records_np[i])
    records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
    tic = time.time()
    first_position, last_position =  bfh.insert_records(records_dict)
    toc = time.time()
    np_insert_time = toc - tic
    print(f'The time needed to insert {num_rows} is {np_insert_time}')
    ######################################################################
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
    #####################################################
    # print first sector records_np
    # print('first sector from records_np:')
    # for row in records_np[:_len]:
    #     print(row)
    #####################################################
    # insert second pack
    records_np = np.concatenate([records_np, np.random.random((num_rows, vec_size))])
    records_dict = [{"id": i + _len, "embed": list(row)} for i, row in enumerate(records_np[_len:])]
    _len = len(records_np)
    tic = time.time()
    first_position2, last_position2 =  bfh.insert_records(records_dict)
    toc = time.time()
    np_insert_time = toc - tic
    print(f'The time needed to insert {num_rows} is {np_insert_time}')
    # get the first sector by get range
    first_sector = bfh.read_positions_in_range(first_position, last_position)
    # print first sector
    # print('first sector from bin file:')
    # for row in first_sector:
    #     print(row[1:])
    # compare the first sector and the same sector from records_np
    if np.allclose(first_sector[:,1:],records_np[:num_rows]):
        print('all first sector are equal')
    else:
        print('first sector are not equal')
    #####################################################
    # print second sector records_np
    # print('second sector from records_np:')
    # for row in records_np[num_rows:]:
    #     print(row) 
    # get the second sector by get range
    second_sector = bfh.read_positions_in_range(first_position2, last_position2)
    # print second sector
    # print('second sector from bin file:')
    # for row in second_sector:
    #     print(row[1:])
    # compare the second sector and the same sector from retreved all
    if np.allclose(second_sector[:,1:],records_np[num_rows:]):
        print('all second sector are equal')
    else:
        print('second sector are not equal')
    #####################################################
    # insert the positions we got in a file
    # define instance of class
    pos_file_path = "positions.bin"
    # empty file if exists
    open(pos_file_path, 'w').close()
    bfh_pos = BinaryFile(pos_file_path)
    bfh_pos.insert_position(0,[first_position,last_position])
    bfh_pos.insert_position(1,[first_position2,last_position2])
    # read the positions
    positions = bfh_pos.read_all_positions()
    # print the positions
    for i,pos in enumerate(positions):
        print(f'pos {i} is ',pos)
        # print real value
        print(f'pos {i} real value is ',[first_position, last_position])
    # compare it with the real values
    if np.array_equal(positions[0][1:], [first_position, last_position]):
        print('first position is equal')
    else:
        print('first position is not equal')
    if np.array_equal(positions[1][1:], [first_position2, last_position2]):
        print('second position is equal')
    else:
        print('second position is not equal')
    #####################################################



    

        
if __name__ == '__main__':
    test()