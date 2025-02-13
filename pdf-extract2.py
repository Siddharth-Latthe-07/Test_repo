import fitz  # PyMuPDF
import json

def extract_executive_names(pdf_path):
    """
    Extracts executive names from the PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        list: A list of executive names.
    """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if 'lines' in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        if text == "EXECUTIVES":
                            capture = True
                            continue

                        if capture:
                            if text.isupper():  # Stop if another section starts
                                return executives

                            executives.append(text)

    return executives

def extract_executive_speech(pdf_path, executives):
    """
    Extracts the text spoken by executives throughout the PDF.
    
    Args:
        pdf_path (str): Path to the PDF file.
        executives (list): List of executive names.
    
    Returns:
        dict: Dictionary where keys are executive names and values are their spoken text.
    """
    doc = fitz.open(pdf_path)
    executive_speech = {name: "" for name in executives}
    current_speaker = None

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if 'lines' in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        if text in executives:
                            current_speaker = text
                            continue
                        
                        if current_speaker:
                            executive_speech[current_speaker] += " " + text

    return executive_speech

def save_to_json(data, output_file):
    """
    Saves extracted speech data to a JSON file.
    
    Args:
        data (dict): Extracted speech data.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# Define the PDF file path
pdf_path = "/mnt/data/3Q-24-Earnings-Call-Transcript (1).pdf"
output_json_path = "/mnt/data/executive_speech.json"

# Extract executive names
executive_roles = {"Chief", "Financial", "Officer", "President", "Vice", "Relations", "CEO"}
executives = extract_executive_names(pdf_path)

# Filter out titles from executive names
clean_executives = [" ".join(word for word in name.split() if word not in executive_roles) for name in executives]

# Extract speech
executive_speech = extract_executive_speech(pdf_path, clean_executives)

# Save to JSON
save_to_json(executive_speech, output_json_path)

print(f"Extraction complete. Data saved to {output_json_path}")
