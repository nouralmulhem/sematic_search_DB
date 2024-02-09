# Semantic Search Engine with Vectorized Databases

<img src="https://i.giphy.com/sRFEa8lbeC7zbcIZZR.webp"/>

    This repository contains the code and documentation for a semantic search engine that utilizes vectorized databases. The project's primary focus is on building an efficient indexing system to retrieve the top-k most similar vectors based on a given query vector. It also includes an evaluation of the performance of the implemented indexing system. The system is designed to retrieve information based on vector space embeddings, demonstrating the implementation and usage of a vectorized database in a practical application.

## <img align= center width=50px height=50px src="https://user-images.githubusercontent.com/71986226/154075883-2a5679d2-b411-448f-b423-9565babf35aa.gif"> Table of Contents
- <a href ="#Overview">Overview</a>
- <a href ="#started"> Get Started</a>
- <a href ="#modules"> Modules</a>
- <a href ="#contributors">Contributors</a>
- <a href ="#license">License</a>

## <img align="center"  height =50px src="https://user-images.githubusercontent.com/71986226/154076110-1233d7a8-92c2-4d79-82c1-30e278aa518a.gif"> Project Overview <a id = "Overview"></a>

The key components of the project include:
- `VecDB`: A class representing the vectorized database, responsible for storing and retrieving vectors.
- `insert_records()`: A method to insert vectors into the database.
- `retrieve()`: A method to retrieve the top-k most similar based on a given query vector.
- `_cal_score()`: A helper method to calculate the cosine similarity between two vectors.
- `_build_index()`: The function responsible for the indexing.

## <img  align= center width=50px height=50px src="https://c.tenor.com/HgX89Yku5V4AAAAi/to-the-moon.gif"> Get Started <a id = "started"></a>

To get started with the project, follow these steps:

1. Clone the repository to your local machine.
2. Run the provided code, you can see the notebook for more clarification.
3. Customize the code and add any additional features as needed.
4. Run the evaluation to assess the accuracy of your implementation.

## <img  align= center width=50px height=50px src="https://cdn.pixabay.com/animation/2022/07/31/06/27/06-27-17-124_512.gif"> Modules <a id ="modules"></a>

The project provides a `VecDB` class that you can use to interact with the vectorized database. Here's an example of how to use it:

```python
from VecDB import VecDB

# Create an instance of VecDB
db = VecDB()

# Insert records into the database
records = [
    {
        "id": 1,
        "embed": [0.1, 0.2, 0.3, ...]  # Image vector of dimension 70
    },
    {
        "id": 2,
        "embed": [0.4, 0.5, 0.6, ...]
    },
    ...
]
db.insert_records(records)

# Retrieve similar images for a given query
query_vector = [0.7, 0.8, 0.9, ...]  # Query vector of dimension 70
similar_images = db.retrieve(query_vector, top_k=5)
print(similar_images)
```

The project also provides a `BinaryFile` class that you can use to read and write binary files. Here's an example of how to use it:

```python
# setup data
num_rows = 1000000
vec_size = 70
# define instance of class
file_path = "data.bin"
# empty file if exists
open(file_path, 'w').close()
bfh = BinaryFile(file_path)
# create data and write to binary file
records_np = np.random.random((num_rows, vec_size))
records_dict = [{"id": i, "embed": list(row)} for i, row in enumerate(records_np)]
bfh.insert_records(records_dict)
# read and verify a single record
random_row_id = random.randint(0, num_rows - 1)
vec_ran = bfh.read_row(random_row_id)[1:]
vec_real = records_np[random_row_id]
print('Single record verification:', np.allclose(vec_ran, vec_real))
# read all records and verify
retrieved_all = bfh.read_all()
retrieved_all = retrieved_all[:,1:]  # remove id from retrieved all
print('All records verification:', np.allclose(retrieved_all, records_np))
```

<hr style="background-color: #4b4c60"></hr>
<a id ="Contributors"></a>

## <img align="center"  height =60px src="https://user-images.githubusercontent.com/63050133/156777293-72a6e681-2582-4a9d-ad92-09d1181d47c7.gif"> Contributors <a id ="contributors"></a>

<br>
<table >
  <tr>
        <td align="center"><a href="https://github.com/Ahmed-H300"><img src="https://avatars.githubusercontent.com/u/67925988?v=4" width="150px;" alt=""/><br /><sub><b>Ahmed Hany</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/nouralmulhem"><img src=https://avatars.githubusercontent.com/u/76218033?v=4" width="150px;" alt=""/><br /><sub><b>Nour Ziad</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/Mohabz-911"><img src=https://avatars.githubusercontent.com/u/68201932?v=4" width="150px;" alt=""/><br /><sub><b>Mohab Zaghloul</b></sub></a><br /></td>
        <td align="center"><a href="https://github.com/Fathi79"><img src=https://avatars.githubusercontent.com/u/96377553?v=4" width="150px;" alt=""/><br /><sub><b>Abdelrhman M.Fathy</b></sub></a><br /></td>
  </tr>
</table>

<hr style="background-color: #4b4c60"></hr>

<a id ="License"></a>

## 🔒 License <a id ="license"></a>

> **Note**: This software is licensed under MIT License, See [License](https://github.com/nouralmulhem/sematic_search_DB/blob/main/LICENSE).

