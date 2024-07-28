import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

# Load job descriptions and vectors
with open('jobs.json', 'r') as f:
    jobs = json.load(f)

job_descriptions = [job['description'] for job in jobs]
job_vectors = [job['embedding'] for job in jobs]

# Load resume text and embedding
with open('resume.json', 'r') as f:
    resume = json.load(f)

    
resume_text = resume[0]['resume']    
resume_embedding = np.array(resume[0]['embedding'])

# BM25 Keyword Search
def bm25_search(resume, job_list):
    tokenized_docs = [job.split() for job in job_list]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = resume.split()
    scores = bm25.get_scores(tokenized_query)
    top_n = np.argsort(scores)[::-1][:10]
    return top_n, scores

# Cosine Similarity Vector Search
def vector_search(resume_vector, job_vectors):
    similarities = cosine_similarity([resume_vector], job_vectors)[0]
    top_n = np.argsort(similarities)[::-1][:10]
    return top_n, similarities

# Reciprocal Rank Fusion
def reciprocal_rank_fusion(ranked_lists, k=60):
    scores = {}
    for ranked_list in ranked_lists:
        for rank, index in enumerate(ranked_list):
            if index not in scores:
                scores[index] = 0
            scores[index] += 1 / (k + rank + 1)
    # Sort by combined score in descending order
    final_ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return final_ranking

# Perform BM25 search
bm25_top_indices, bm25_scores = bm25_search(resume_text, job_descriptions)

# Perform Vector search
vector_top_indices, vector_scores = vector_search(resume_embedding, job_vectors)

# Combine rankings using Reciprocal Rank Fusion
combined_ranking = reciprocal_rank_fusion([bm25_top_indices, vector_top_indices])

# Print results
print("Top 5 BM25 matches:")
for index in bm25_top_indices:
    print(f"Job ID: {jobs[index]['title']} {jobs[index]['id']}, Score: {bm25_scores[index]}")

print("\nTop 5 Vector matches:")
for index in vector_top_indices:
    print(f"Job ID: {jobs[index]['title']} {jobs[index]['id']}, Score: {vector_scores[index]}")

print("\nTop matches using Reciprocal Rank Fusion:")
for index, score in combined_ranking[:5]:
    print(f"Job ID: {jobs[index]['title']} {jobs[index]['id']}, Combined Score: {score}")
    
