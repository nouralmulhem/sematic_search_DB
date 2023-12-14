import numpy as np
from nanopq import PQ
import time

# Number of training records
N = 1_000_000
# Dimension of each record
D = 70

# Generate some random data for demonstration
np.random.seed(42)
data = np.random.rand(N, D).astype(np.float32)
print(data)


# Set the number of subquantizers and subquantizer size
M, Ks = 70, 256

# Initialize the Product Quantizer
pq = PQ(M=M, Ks=Ks)

# Train the Product Quantizer on your data
start_time = time.time()
pq.fit(data, iter=3)
end_time = time.time()

# Calculate the training time
training_time = end_time - start_time
print(f"Training time: {training_time} seconds")

# Encode your data using the trained Product Quantizer
start_time = time.time()
codes = pq.encode(data)
end_time = time.time()

# Calculate the encoding time
encoding_time = end_time - start_time
print(f"Encoding time: {encoding_time} seconds")

print(codes)
print(codes.shape[0])

# Decode the codes to get the approximated data
decoded_data = pq.decode(codes)

# Print the reconstruction error
reconstruction_error = np.mean(np.linalg.norm(data - decoded_data, axis=1))
print(f"Reconstruction error: {reconstruction_error}")


