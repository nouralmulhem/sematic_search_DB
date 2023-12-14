import numpy as np
from includes import CreateDatabase, BinaryFile
from nanopq import PQ
import pickle
import time

is_new_db = False
is_sampled = False

# Create the database
db = CreateDatabase(file_path="saved_db.bin", new_db=is_new_db)

if is_new_db:
    for iter in range(2000):
        records_np = np.random.random((10_000, 70))
        records_dict = [{"id": i + (iter*10_000), "embed": list(row)}
                        for i, row in enumerate(records_np)]
        db.insert_records(records_dict)

# Now let's create the sampled DB
sampled_db = CreateDatabase(file_path='sampled_db.bin', new_db=is_sampled)


if is_sampled:
    to_be_inserted = []
    for iter in range(1_000_000):
        print('Iteration: '+str(iter))
        #   Read 20 records. Get their mean
        section = db.bfh.read_records(iter*20,(iter+1)*20)
        section = np.array([row[1:] for row in section], dtype=np.float32)
        sample = list(np.mean(section, 0))
        # Append to the list that will be inserted into the sampled DB
        to_be_inserted.append({"id":iter, "embed":sample})
        # Clear
        section = []
        sample = []
    
    sampled_db.insert_records(to_be_inserted)


start_time = time.time()
# Number of training records
N = 1_000_000
# Dimension of each record
D = 70

# Load the sampled records
data = sampled_db.bfh.read_all()
assert data.shape == (1000000, 71)
data = np.delete(data, 0, axis=1)
assert data.shape == (1000000, 70)
assert data.dtype == np.float32


# Set the number of subquantizers and subquantizer size
M, Ks = 70, 256

# Initialize the Product Quantizer
pq = PQ(M=M, Ks=Ks)

# Train the Product Quantizer on your data
pq.fit(data, iter=3)

# Encode your data using the trained Product Quantizer
codes = pq.encode(data)

print(codes)

# Save the pq instance in a pickle file
with open('pq.pkl', 'wb') as f:
    pickle.dump(pq, f)

# Save the codewords in a pickle file
with open('pq_codes.pkl', 'wb') as f:
    pickle.dump(codes, f)

end_time = time.time()
print("Total time = " + str(end_time-start_time) + ' seconds')

# # Decode the codes to get the approximated data
# decoded_data = pq.decode(codes)

# # Print the reconstruction error
# reconstruction_error = np.mean(np.linalg.norm(data - decoded_data, axis=1))
# print(f"Reconstruction error: {reconstruction_error}")