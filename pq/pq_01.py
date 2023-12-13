import numpy as np
from nanopq import PQ
import time

# Generate some random data for demonstration
np.random.seed(42)
data = np.random.rand(1_000_000, 70).astype(np.float32)


# Set the number of subquantizers and subquantizer size
M, Ks = 10, 128

# Initialize the Product Quantizer
pq = PQ(M=M, Ks=Ks)

# Train the Product Quantizer on your data
start_time = time.time()
pq.fit(data)
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

# Decode the codes to get the approximated data
decoded_data = pq.decode(codes)

# Print the reconstruction error
reconstruction_error = np.mean(np.linalg.norm(data - decoded_data, axis=1))
print(f"Reconstruction error: {reconstruction_error}")


