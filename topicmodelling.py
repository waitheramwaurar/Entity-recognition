import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Loading the data
df = pd.read_csv("text_data - text_data.csv")
text_data = df['story'].dropna().astype(str)

# Preprocess the text using spaCy
nlp = spacy.load("en_core_web_sm")

def preprocess(doc):
    doc = nlp(doc)
    return " ".join([token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop])

cleaned_text = text_data.apply(preprocess)

# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(cleaned_text)

# Fit the LDA model
lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
lda_model.fit(dtm)

# Display top words in each topic
words = vectorizer.get_feature_names_out()
topics = {}
for i, topic in enumerate(lda_model.components_):
    topic_words = [words[i] for i in topic.argsort()[-10:]]
    topics[f"Topic #{i + 1}"] = topic_words

# Display the topics
print(topics)
