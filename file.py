import decimal
import random
import struct

import numpy as np


class BinaryFile:
    def __init__(self, filename):
        self.filename = filename
        self.vec_size = 70
        self.length = 24
        self.id_size = 4

    def insert_row(self, row_id, row_data):
        with open(self.filename, 'ab') as file:
            # Pack the ID into a binary format
            packed_id = struct.pack('i', row_id)
            # Write the packed ID to the file
            file.write(packed_id)
            # Convert each numpy.float64 to a fixed-length string and write it to the file
            for x in row_data:
                str_x = str(decimal.Decimal(str(x))).ljust(self.length)  # Pad the string with spaces to a length of self.length
                file.write(str_x.encode())

    def read_row(self, row_id):
        with open(self.filename, 'rb') as file:
            # Calculate the position of the row
            position = row_id * (self.id_size + self.vec_size * self.length)  # Size of one row (ID + self.vec_size * numpy.float64 values)
            # Seek to the position of the row
            file.seek(position)
            # Read the ID
            packed_id = file.read(self.id_size)
            row_id = struct.unpack('i', packed_id)[0]
            # Read each numpy.float64
            row_data = []
            for _ in range(self.vec_size):
                str_x = file.read(self.length).decode().strip()  # Remove the padding spaces
                x = decimal.Decimal(str_x)
                row_data.append(x)
            row_data = np.array([float(d) for d in row_data])
            return row_data


def test():
    # setup data
    num_rows = 1000000
    num_tests = 1000
    vec_size = 70
    # define instance of class
    file_path = "data.bin"
    # empty file if exists
    open(file_path, 'w').close()
    bfh = BinaryFile(file_path)
    # create data and write to binary file
    records_np = np.random.random((num_rows, vec_size))
    for i in range(num_rows):
        bfh.insert_row(i,records_np[i])

    count_test = 0
    failed_ids = []
    for i in range(num_tests):
        random_row_id = random.randint(0, (num_tests - 1))
        vec_ran = bfh.read_row(random_row_id)
        vec_real = records_np[random_row_id]
        # compare the 2 vectors if one is not equal break
        if np.array_equal(vec_ran,vec_real):
            count_test += 1
        else:
            failed_ids.append(random_row_id)

    print(f"Passed {count_test} tests")
    print(f"Failed {len(failed_ids)} tests")
    print(f"Failed ids: {failed_ids}")


    # uncomment to print the vector
    # print(records_np[1])
    # print('---------------------')
    # print(bfh.read_row(1))
    # if np.array_equal(records_np[1],bfh.read_row(1)):
    #         print('-----------eq----------')

    # else:
    #        print('-----------noteq----------')

        
if __name__ == '__main__':
    test()