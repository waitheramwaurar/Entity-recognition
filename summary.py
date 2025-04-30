import pandas as pd
from transformers import pipeline

# Load the dataset
df = pd.read_csv("text_data - text_data.csv")

from transformers import pipeline

# Load the pre-trained summarization model from HuggingFace
summarizer = pipeline("summarization")

# Define a function to summarize long texts by splitting into smaller chunks
def summarize_text(text, max_length=1024, summary_max_length=150):
    # Split the text into chunks that fit the model's maximum token length
    words = text.split()
    chunk_size = max_length // 2  # Approximate size of each chunk (adjust as necessary)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    
    # Summarize each chunk
    summaries = []
    for chunk in chunks:
        chunk_text = " ".join(chunk)
        summary = summarizer(chunk_text, max_length=summary_max_length, min_length=50, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    # Combine all summaries
    final_summary = " ".join(summaries)
    return final_summary

# Use the first article from the dataset as a sample
sample_story = df['story'].iloc[0]  # Modify if you want to summarize a different story

# Summarize the text
summary = summarize_text(sample_story)

# Print the summary
print(summary)
