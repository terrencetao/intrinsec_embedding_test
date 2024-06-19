import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
import csv
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import argparse


def load_word_vectors(word_vectors_file_path):
    if not os.path.exists(word_vectors_file_path):
        print(f"Error: Word vectors file '{word_vectors_file_path}' not found.")
        return None
    with open(word_vectors_file_path, 'rb') as file:
        return pickle.load(file)


def get_txt_file(path):
  return [os.path.join(path, filename) for filename in os.listdir(path)  if filename.endswith('.txt')]

    
    
    
    
#=====================================================================================================================================================================================================
# Test 1: Analogies Evaluation
#====================================================================================================================================================================================================
# Load analogy questions
def load_analogy_questions(directory_path):
   analogy_questions = []
   for filepath in get_txt_file(directory_path):
      with open(filepath, 'r') as file:
         for line in file:
            analogy = line.strip().split()
            if len(analogy) == 4:
               analogy_questions.append(analogy)
   return analogy_questions

def evaluate_analogies(word_vectors, analogy_questions):
    correct_count = 0
    total_count = len(analogy_questions)

    for analogy in analogy_questions:
       A, B, C, expected_D = analogy
       try:
           A_vector = word_vectors[A]
           B_vector = word_vectors[B]
           C_vector = word_vectors[C]
                
           calculated_D_vector = B_vector - A_vector + C_vector
           calculated_D_vector = calculated_D_vector.reshape(1, -1)
                
           word_vectors_array = np.array([word_vectors[word] for word in word_vectors if word != C])
           similarities = cosine_similarity(word_vectors_array, calculated_D_vector)
                
           closest_word_index = np.argmax(similarities)
           closest_word = list(word_vectors.keys())[closest_word_index]
                
           if closest_word == expected_D:
               correct_count += 1
       except KeyError:
             pass

    accuracy = correct_count / total_count * 100
    return accuracy
        
def analogies_evaluation(word_to_vec, dataset_file_path):
    

    
    analogy_questions = load_analogy_questions(dataset_file_path)
    accuracy = evaluate_analogies(word_to_vec, analogy_questions)
    return accuracy
    
    
    
    
    
    
    
    
    
    
    
#===================================================================================================================================================================================================
# Test 2: Outlier Detection
#==============================================================================================================================================================================================

 # Define a method to compute the embedding vector for a list of words
def compute_embedding(word):
    if word in word_to_vec:
       return word_to_vec[word]
    else:
       return None

    # Define a method to detect outliers based on embedding vectors
def detect_outliers(embeddings):
    centroid = np.mean(embeddings, axis=0)
    distances = cosine_distances(embeddings, [centroid])
    threshold = np.percentile(distances, 60)
    outlier_labels = [1 if distance > threshold else 0 for distance in distances]
    return outlier_labels


def outlier_detection(word_to_vec, dataset_folder):
   
    y_true = []
    y_pred = []
    for file_path in get_txt_file(dataset_folder):
            with open(file_path, 'r') as file:
                words = []
                outliers = []
                read_outliers = False
                for line in file:
                    word = line.strip()
                    if not word:
                        read_outliers = True
                        continue
                    if read_outliers:
                        if compute_embedding(word) is not None:
                            outliers.append(word)
                    else:
                        if compute_embedding(word) is not None:
                            words.append(word)

                word_embeddings = [compute_embedding(word) for word in words]
                outlier_embeddings = [compute_embedding(word) for word in outliers]
                word_embeddings = [emb for emb in word_embeddings if emb is not None]
                outlier_embeddings = [emb for emb in outlier_embeddings if emb is not None]

                if word_embeddings and outlier_embeddings:
                    embeddings = np.concatenate([np.array(word_embeddings), np.array(outlier_embeddings)])
                    predicted_labels = detect_outliers(embeddings)
                    y_true.extend([1] * len(words) + [0] * len(outliers))
                    y_pred.extend(predicted_labels)

    accuracy = accuracy_score(y_true, y_pred) * 100
    return accuracy
    
    
    
#==================================================================================================================================================================================================
# Test 3: Clustering
#==================================================================================================================================================================================================


def normalize_embeddings(embeddings):
    # Normalize embeddings and handle NaN or infinite values
    normalized_embeddings = np.array(embeddings)
    normalized_embeddings /= np.linalg.norm(normalized_embeddings, axis=1, keepdims=True)
    normalized_embeddings = np.nan_to_num(normalized_embeddings)
    return normalized_embeddings

def compute_purity(cluster_labels, true_labels):
    total_count = len(cluster_labels)
    cluster_categories = {}
    
    # Create a dictionary to store the counts of true labels in each cluster
    for cluster_label, true_label in zip(cluster_labels, true_labels):
        if cluster_label not in cluster_categories:
            cluster_categories[cluster_label] = {}
        if true_label not in cluster_categories[cluster_label]:
            cluster_categories[cluster_label][true_label] = 0
        cluster_categories[cluster_label][true_label] += 1
    
    # Compute purity
    purity = 0
    for cluster_label, categories in cluster_categories.items():
        majority_count = max(categories.values())
        purity += majority_count
    
    return purity / total_count



def clustering(word_to_vec, directory, embedding_dim):
    purity_values = []
    silhouette_values = []
    
    for _ in range(100):
        for file_path in get_txt_file(directory):
                data = []
                with open(file_path, 'r') as file:
                    next(file)
                    for line in file:
                        word, category, _, _ = line.strip().split()
                        data.append((word, category))

                words = [entry[0] for entry in data]
                categories = [entry[1] for entry in data]

                word_embeddings = []
                for word in words:
                    if word in word_to_vec:
                        word_embeddings.append(word_to_vec[word])
                    else:
                        word_embeddings.append(np.zeros(embedding_dim))

                normalized_embeddings = normalize_embeddings(word_embeddings)

                num_clusters = len(set(categories))
                kmeans = KMeans(n_clusters=num_clusters)
                cluster_labels = kmeans.fit_predict(normalized_embeddings)

                purity = compute_purity(cluster_labels, categories)
                silhouette_avg = silhouette_score(word_embeddings, cluster_labels)
                purity_values.append(purity)
                silhouette_values.append(silhouette_avg)

    mean_purity = np.mean(purity_values)
    mean_silhouette = np.mean(silhouette_values)
    return mean_purity, mean_silhouette
    
    

        
        
        
        
        
        
        
#=====================================================================================================================================================================================
## Test 4: Similarity  
#====================================================================================================================================================================================================
def load_word_similarity_data(dataset_file_path, word_vectors):
    if not os.path.exists(dataset_file_path):
        print(f"Error: Word similarity dataset file '{dataset_file_path}' not found.")
        return None

    # Read word similarity dataset
    word_similarity_data = []
    with open(dataset_file_path, 'r') as file:
        for line in file:
            word1, word2, score = line.strip().split()
            if word1 in word_vectors and word2 in word_vectors:
                word_similarity_data.append((word1, word2, float(score)))
    return word_similarity_data

def calculate_similarity(word_vectors, word_pairs):
    similarity_scores = []
    for word1, word2 in word_pairs:
        similarity_scores.append(1-cosine(word_vectors[word1], word_vectors[word2]))
    return similarity_scores

def compute_spearman_correlation(computed_scores, human_scores):
    correlation_coefficient, _ = spearmanr(computed_scores, human_scores)
    return correlation_coefficient

def optimize_and_run_similarity_test(word_vectors, dataset_file_paths):
    # Load word vectors
    
    if word_vectors is None:
        return None
    coef_corr_value = []
    for dataset_file_path in dataset_file_paths:
        # Load word similarity data
        word_similarity_data = load_word_similarity_data(dataset_file_path, word_vectors)
        if not word_similarity_data:
            continue

        # Extract word pairs and human similarity scores
        word_pairs = [(word1, word2) for word1, word2, _ in word_similarity_data]
        human_similarity_scores = [score for _, _, score in word_similarity_data]

        # Calculate computed similarity scores
        computed_similarity_scores = calculate_similarity(word_vectors, word_pairs)

        # Compute Spearman correlation
        correlation_coefficient = compute_spearman_correlation(computed_similarity_scores, human_similarity_scores)
        coef_corr_value.append(correlation_coefficient)
        # Print results
        print(f"Spearman Correlation: {correlation_coefficient} for dataset {dataset_file_path}")
     
    return np.mean(coef_corr_value)






















# Combine the results into a CSV file
def write_results_to_csv(w2v,analogies_result, outlier_result, clustering_result, sim_result):
    with open(w2v+'test_results.csv', 'w', newline='') as csvfile:
        fieldnames = ['Test', 'Result']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'Test': 'Analogies Evaluation', 'Result': analogies_result})
        writer.writerow({'Test': 'Outlier Detection', 'Result': outlier_result})
        writer.writerow({'Test': 'Clustering Purity', 'Result': clustering_result[0]})
        writer.writerow({'Test': 'Clustering Silhouette', 'Result': clustering_result[1]})
        writer.writerow({'Test': 'Similarity Evaluation', 'Result': sim_result})

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_folder', help ='source folder')
    
    parser.add_argument('--w2v', help ='source for word2vec pk file')
    parser.add_argument('--dim', help ='dimension of embedding')
    args = parser.parse_args()
    model_folder = args.model_folder
    w2v_file = args.w2v
    embedding_dim = int(args.dim)
    
    word_vectors_file_path = os.path.join(model_folder, w2v_file)
    word_to_vec = load_word_vectors(word_vectors_file_path)
    
    
    # Execute the three tests
    sim_result = optimize_and_run_similarity_test(word_to_vec, dataset_file_paths=get_txt_file('similarity'))
    outlier_result = outlier_detection(word_to_vec,'outlier')
    clustering_result = clustering(word_to_vec,'category', embedding_dim)
    
    analogies_result = analogies_evaluation(word_to_vec,'analogy')

    # Write the results to a CSV file
    write_results_to_csv(w2v_file, analogies_result, outlier_result, clustering_result, sim_result)

