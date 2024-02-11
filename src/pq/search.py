import numpy as np
from includes import CreateDatabase, BinaryFile, cal_score
from nanopq import PQ
import pickle
import time
import random

# Time to make a query :)
# Open the database
db = CreateDatabase(file_path="saved_db.bin", new_db=False)
# Choose a random vector from the DB to be the query
random_id = random.randint(0,20_000_000-21)
record = db.bfh.read_record(random_id)
print(random_id)
print(record)

hidden_id = record[0]       # Kept hidden till the search is complete
query = np.array(record[1:], dtype=np.float32)


# Load the pq instance and it's codewords
pq:PQ
with open('pq.pkl', 'rb') as f:
    pq = pickle.load(f)

pq_codes:np.ndarray
with open('pq_codes.pkl', 'rb') as f:
    pq_codes = pickle.load(f)

# Get the distance table
dt = pq.dtable(query=query)

# Get the distances vector
dists = dt.adist(codes=pq_codes)

# Append ids to the pq_codes
pq_codes = np.array([[i] + pq_code for i, pq_code in enumerate(pq_codes)], dtype=np.uint16)

# Append distances vector to pq_codes
pq_codes = np.concatenate((dists.reshape(-1, 1), pq_codes), axis=1)

# Sort the codes
# If two codes have the same distance, the smaller id is chosen first
pq_codes.sort()

# Chose the top 10 codes
pq_codes = pq_codes[:10]
print(pq_codes)
candidate_ids = set([int(x) for x in pq_codes[:,1]])    # Stored in a set to exclude repititive ids

# Get the sections of the candidate ids
candidates = []
for id in candidate_ids:
    print('Section: '+str(id))
    #   Read 20 records. Get their mean
    section = db.bfh.read_records(id*20,(id+1)*20)
    print(section)
    # section = np.array([row[1:] for row in section], dtype=np.float32)
    # Append to the list that will be inserted into the sampled DB
    candidates.extend(section)
    # Clear
    section = []


# Dot product with the candidates
scores = []
for row in candidates:
    id = int(row[0])
    embed = [float(e) for e in row[1:]]
    score = cal_score(query, embed)
    scores.append((score, id))
# here we assume that if two rows have the same score, return the lowest ID
# scores = sorted(scores, reverse=True)
scores.sort(reverse=True)
print(scores)
# print (scores[:][1])
# print (hidden_id)
