import spacy
from collections import Counter
import matplotlib.pyplot as plt

# Loading the dataset
import pandas as pd
text_df = pd.read_csv("text_data - text_data.csv")

# Loading spaCy NLP library's English model
nlp = spacy.load("en_core_web_sm")

# Combine all stories into one text
all_text = " ".join(text_df["story"].dropna())

# Apply the NLP model
doc = nlp(all_text)

# Define entity labels of interest
labels_of_interest = {
    "GPE": "Geographical Locations",
    "ORG": "Organizations",
    "PERSON": "People",
    "DATE": "Dates",
    "MONEY": "Amounts"
}

# Extract entities by category
entities = {label: [] for label in labels_of_interest.values()}
for entity in doc.ents:
    if entity.label_ in labels_of_interest:
        entities[labels_of_interest[entity.label_]].append(entity.text)

# Count and visualize top 5 entities in each category
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.flatten()

for i, (category, values) in enumerate(entities.items()):
    top_entities = Counter(values).most_common(5)
    labels, counts = zip(*top_entities) if top_entities else ([], [])
    axes[i].barh(labels, counts, color="skyblue")
    axes[i].set_title(f"Top 5 {category}")
    axes[i].invert_yaxis()

# Hide the unused subplot if odd number
axes[-1].axis('off')
plt.tight_layout()
plt.show()
