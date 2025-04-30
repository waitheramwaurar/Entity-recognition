import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("text_data - text_data.csv")

# Clean the text data
def basic_preprocess(doc):
    # Remove non-alphabetic characters and convert to lowercase
    doc = re.sub(r'[^a-zA-Z\s]', '', doc)
    doc = doc.lower()
    return doc

# Apply preprocessing
df['cleaned_text'] = df['story'].apply(basic_preprocess)

# Vectorize the text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)
X = vectorizer.fit_transform(df['cleaned_text'])

# Apply KMeans clustering
num_categories = 10  # Number of predefined categories
kmeans = KMeans(n_clusters=num_categories, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Define the category labels (your predefined categories)
categories = ['Advanced Gene Editing', 'mRNA Technology', 'CAR-T Cell Therapy', 'Organoids and Tissue Engineering',
              'Single-Cell Genomics', 'Synthetic Biology', 'Biological Computing', 'Wearable Biosensors', 
              'Microbiome Therapeutics', 'Nanomedicine']

# Encode the clusters with category labels
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['cluster'])

# Map the cluster labels to the category names
df['category_name'] = df['category'].apply(lambda x: categories[x])

# Calculate the ratio of each category
category_counts = df['category_name'].value_counts(normalize=True)  # Normalized to get the ratio
category_ratios = category_counts.to_dict()

# Display the results
print(category_ratios)
print(df[['story', 'category_name']].head())  # Display the first few stories with their assigned categories
