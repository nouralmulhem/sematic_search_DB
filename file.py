import random
import struct
import numpy as np

class BinaryFile:
    def __init__(self, filename):
        self.filename = filename

    def insert_row(self, row_id, row_data):
        with open(self.filename, 'ab') as file:
            # Pack the ID and the float values into a binary format
            packed_data = struct.pack('i5f', row_id, *row_data)
            # Write the packed data to the file
            file.write(packed_data)

    def read_row(self, row_id):
        with open(self.filename, 'rb') as file:
            # Calculate the position of the row
            position = row_id * (4 + 5 * 4)  # Size of one row (ID + 5 floats)
            # Seek to the position of the row
            file.seek(position)
            # Read the row
            packed_data = file.read(4 + 5 * 4)  # Size of one row (ID + 5 floats)
            data = struct.unpack('i5f', packed_data)
            return np.array(data[1:])

def test():
    # setup data
    num_rows = 10000
    num_tests = 100
    # define instance of class
    file_path = "data.bin"
    # empty file if exists
    open(file_path, 'w').close()
    bfh = BinaryFile(file_path)
    # create data and write to binary file
    records_np = np.random.random((num_rows, 5))
    for i in range(num_rows):
        bfh.insert_row(i,records_np[i])

    print(records_np[0])
    print('---------------------')
    print(bfh.read_row(0))
    if np.array_equal(records_np[0],bfh.read_row(0)):
            print('-----------eq----------')

    else:
           print('-----------noteq----------')

    # count_test = 0
    # failed_ids = []
    # for i in range(num_tests):
    #     random_row_id = random.randint(0, (num_tests - 1))
    #     vec_ran = bfh.read_row(random_row_id)
    #     vec_real = records_np[random_row_id]
    #     # compare the 2 vectors if one is not equal break
    #     if np.array_equal(vec_ran,vec_real):
    #         count_test += 1
    #     else:
    #         failed_ids.append(random_row_id)

    # print(f"Passed {count_test} tests")
    # print(f"Failed {len(failed_ids)} tests")
    # print(f"Failed ids: {failed_ids}")

        
if __name__ == '__main__':
    test()