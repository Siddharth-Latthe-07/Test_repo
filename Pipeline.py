import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure you have the stopwords package
nltk.download("stopwords")
nltk.download("punkt")

# Define custom stopwords (including NLTK's English stopwords)
custom_stopwords = set(stopwords.words("english")).union({
    "please", "u", "ha", "wa", "r", "2", "dont", "dear", "sir", "pls", "plz", "see",
    "ji", "hai", "must", "man", "tell", "mr", "ki", "ho", "aur", "ppls", "fu",
    "nahi", "denge", "madam", "surely", "well", "br", "mh", "ye", "didnt", "dr",
    "&", "thank you", "good", "morning", "--", "q&a", "?", "$", "thank you",
    "thank much", "indiscernible"
})

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):  # Handle non-string values
        return ""

    # Remove unwanted characters (digits, special chars, file extensions, etc.)
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = re.sub(r"\b\d+\b", "", text)  # Remove numbers
    text = re.sub(r"\b[a-zA-Z]\b", "", text)  # Remove single-letter words
    text = re.sub(r"\b(?:xlsx|py|import|from|re|yml|le)\b", "", text, flags=re.IGNORECASE)  # Remove file types & keywords

    # Tokenization
    words = word_tokenize(text.lower())

    # Remove stopwords
    words = [word for word in words if word not in custom_stopwords]

    # Reconstruct sentence and remove short sentences
    cleaned_text = " ".join(words)
    return cleaned_text if len(cleaned_text.split()) >= 6 else ""

# Sample DataFrame
data = {
    "openingtext": [
        "195 import nitk",
        "tokenizer AutoTokenizer.from_pretrained('bert-base-uncased')",
        "hello sir good morning",
        "this is an example sentence which should be cleaned properly",
        "re yml xlsx Py M",
        "this is a valid sentence that should remain in the dataset"
    ]
}

df = pd.DataFrame(data)

# Apply preprocessing
df["cleaned_text"] = df["openingtext"].apply(clean_text)

# Remove empty entries after cleaning
df = df[df["cleaned_text"] != ""]

print(df)
