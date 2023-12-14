import numpy as np
from includes import CreateDatabase, BinaryFile

is_new_db = True
is_sampled = True

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
    for iter in range(1_000_000):
        print('Iteration: '+str(iter))
        #   Read 20 records. Get their mean
        section = db.bfh.read_records(iter*20,(iter+1)*20)
        section = np.array([row[1:] for row in section])
        sample = list(np.mean(section, 0))
        # Store that into the sampled DB
        sampled_db.bfh.insert_row(iter, sample)
        # sampled_db.insert_records([{"id":iter,"embed":sample}])
        # Clear
        section = []
        sample = []
