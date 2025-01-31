import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
english_stopwords = set(stopwords.words('english'))

def is_hindi(text):
    """Check if a line contains Hindi characters using Unicode range."""
    return bool(re.search(r'[\u0900-\u097F]', text))

def clean_text(text):
    """Remove symbols and stop words from English sentences."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove symbols and special characters
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in english_stopwords]  # Remove stopwords
    return " ".join(filtered_words)

def process_sheet(input_path, output_path, sheet_name):
    """Process a specific sheet in the Excel file."""
    df = pd.read_excel(input_path, sheet_name=sheet_name, engine='openpyxl')

    if 'description' in df.columns:
        df = df.dropna(subset=['description'])  # Drop rows with NaN descriptions
        df['description'] = df['description'].astype(str)  # Ensure text format

        # Remove Hindi lines
        df = df[~df['description'].apply(is_hindi)]

        # Clean English text
        df['description'] = df['description'].apply(clean_text)

    # Save the cleaned sheet back to a new Excel file
    df.to_excel(output_path, sheet_name=sheet_name, index=False, engine='openpyxl')

# Example Usage
input_file = "input.xlsx"  # Change this to your actual file
output_file = "cleaned_output.xlsx"
sheet_name = "Sheet1"  # Change this to your actual sheet name

process_sheet(input_file, output_file, sheet_name)

print("Processing complete. Cleaned file saved as:", output_file)







import pandas as pd
import re
import spacy

# Load spaCy's English model
nlp = spacy.load("en_core_web_sm")
english_stopwords = nlp.Defaults.stop_words  # Get spaCy's stop words list

def is_hindi(text):
    """Check if a line contains Hindi characters using Unicode range."""
    return bool(re.search(r'[\u0900-\u097F]', text))

def clean_text(text):
    """Remove symbols and stop words from English sentences."""
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in english_stopwords]  # Remove stopwords
    return " ".join(filtered_words)

def process_sheet(input_path, output_path, sheet_name):
    """Process a specific sheet in the Excel file."""
    df = pd.read_excel(input_path, sheet_name=sheet_name, engine='openpyxl')

    if 'description' in df.columns:
        df = df.dropna(subset=['description'])  # Drop rows with NaN descriptions
        df['description'] = df['description'].astype(str)  # Ensure text format

        # Remove Hindi lines
        df = df[~df['description'].apply(is_hindi)]

        # Clean English text
        df['description'] = df['description'].apply(clean_text)

    # Save the cleaned sheet back to a new Excel file
    df.to_excel(output_path, sheet_name=sheet_name, index=False, engine='openpyxl')

# Example Usage
input_file = "input.xlsx"  # Change this to your actual file
output_file = "cleaned_output.xlsx"
sheet_name = "Sheet1"  # Change this to your actual sheet name

process_sheet(input_file, output_file, sheet_name)

print("Processing complete. Cleaned file saved as:", output_file)
