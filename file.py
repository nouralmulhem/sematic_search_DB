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
    # define instance of class
    file_path = "data.bin"
    bfh = BinaryFile(file_path)
    # create data and write to binary file
    records_np = np.random.random((5, 5))
    for i in range(5):
        bfh.insert_row(i,records_np[i])
    for r in records_np:
        print(r)
        print('-------------------------------------')

    # read data from binary file
    print('*************************************************')
    print(bfh.read_row(2))
    print('*************************************************')
    for i in range(5):
        print(bfh.read_row(i))
        print('-------------------------------------')



if __name__ == '__main__':
    test()