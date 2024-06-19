import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


# Step 2: Feature Extraction
import numpy as np
import pickle



# Load dataset
data = pd.read_csv('tweet_emotions.csv')

# Load pre-trained word embeddings
with open('word2vec.pk', 'rb') as f:
    word_embeddings = pickle.load(f)

# Calculate the mean vector of all word embeddings in the vocabulary
mean_embedding = np.mean(list(word_embeddings.values()), axis=0)

# Create a function to generate document embeddings from word embeddings
def generate_doc_embedding(text, word_embeddings, mean_embedding):
    words = text.split()
    embedding_dim = len(mean_embedding)
    doc_embedding = np.zeros(embedding_dim)
    word_count = 0
    for word in words:
        if word in word_embeddings:
            doc_embedding += word_embeddings[word]
            word_count += 1
    if word_count > 0:
        doc_embedding /= word_count
    else:
        doc_embedding = mean_embedding
    return doc_embedding

# Generate document embeddings for each text in the dataset
X = np.array([generate_doc_embedding(text, word_embeddings, mean_embedding) for text in data['text']])


# Preprocess text data
# Your preprocessing steps here

# Encode emotions into numerical labels
label_encoder = LabelEncoder()
data['emotion_label'] = label_encoder.fit_transform(data['emotion'])

# Step 2: Feature Extraction
# Convert text data into numerical vectors
vectorizer = TfidfVectorizer(max_features=1000)  # Use TF-IDF as an example
X = vectorizer.fit_transform(data['text'])

# Step 3: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, data['emotion_label'], test_size=0.2, random_state=42)

# Step 4: Model Training
svm_model = SVC(kernel='linear')  # Linear kernel for SVM
svm_model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 6: Hyperparameter Tuning (Optional)
# Grid search or random search for hyperparameter tuning

# Step 7: Predict Emotions
# Use the trained model for predictions on new data
new_text = ["Your new text data here"]
new_text_vectorized = vectorizer.transform(new_text)
predicted_emotion = label_encoder.inverse_transform(svm_model.predict(new_text_vectorized))
print("Predicted Emotion:", predicted_emotion)
