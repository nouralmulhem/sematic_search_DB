import struct
import numpy as np

class BinaryFile:
    def __init__(self, filename):
        self.filename = filename
        self.index = {}  # Maps row ID to position in file

    def insert_row(self, row_id, row_data):
        with open(self.filename, 'ab') as file:
            # Remember the position where we're going to write the data
            position = file.tell()
            # Pack the ID and the float values into a binary format
            packed_data = struct.pack('i5f', row_id, *row_data)
            # Write the packed data to the file
            file.write(packed_data)
            # Update the index
            self.index[row_id] = position
            

    def read_row(self, row_id):
        position = self.index.get(row_id)
        if position is not None:
            with open(self.filename, 'rb') as file:
                # Seek to the position of the row
                file.seek(position)
                # Read the row
                packed_data = file.read(4 + 5 * 4)  # Size of one row (ID + 70 floats)
                data = struct.unpack('i5f', packed_data)
                return np.array(data[1:])
        return None

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