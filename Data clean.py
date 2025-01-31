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











import pandas as pd
import re

# Function to check if a string contains Hindi characters
def contains_hindi(text):
    hindi_pattern = re.compile(r'[\u0900-\u097F]')  # Unicode range for Hindi
    return bool(hindi_pattern.search(text))

# Function to clean the description column
def clean_description(text):
    if pd.isna(text):  # Check if value is NaN
        return text
    # Remove Hindi sentences
    lines = text.split("\n")
    english_lines = [line for line in lines if not contains_hindi(line)]
    cleaned_text = " ".join(english_lines)
    # Remove special characters and numbers
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', cleaned_text)
    return cleaned_text.strip()

# Load Excel file and focus on the 'drad' sheet
file_path = "your_file.xlsx"  # Update with actual file path
df = pd.read_excel(file_path, sheet_name="drad")

# Apply cleaning function to the 'description' column
df['description'] = df['description'].astype(str).apply(clean_description)

# Save the cleaned data back to an Excel file
df.to_excel("cleaned_file.xlsx", sheet_name="drad", index=False)

print("Data cleaning completed. Check 'cleaned_file.xlsx'.")
