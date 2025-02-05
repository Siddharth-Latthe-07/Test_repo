#paragraph by para
import pdfplumber


def format_text(raw_text):
    """
    Cleans and formats raw text extracted from the PDF.
    - Removes excessive whitespace and blank lines.
    - Ensures proper paragraph spacing.
    
    Args:
        raw_text (str): The raw text extracted from the PDF.
    
    Returns:
        str: Formatted and cleaned text.
    """
    # Remove leading/trailing spaces and replace multiple newlines with a single newline
    formatted_text = "\n".join([line.strip() for line in raw_text.splitlines() if line.strip()])
    return formatted_text


def extract_text_without_tables(pdf_path, output_path):
    """
    Extracts all text from a PDF file, excluding text inside tables and organizes by paragraphs.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_path (str): Path to save the formatted text output.
    
    Returns:
        None
    """
    try:
        all_text = []
        
        # Open the PDF using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Get bounding boxes of all tables on the page
                table_bbox = [table.bbox for table in page.find_tables()]
                
                # Extract the full text block
                raw_text = page.extract_text()
                
                # Split text into paragraphs based on multiple newlines
                paragraphs = raw_text.split('\n\n')  # Assuming paragraphs are separated by double newlines
                
                filtered_paragraphs = []
                for paragraph in paragraphs:
                    # Split into lines and filter out lines within table bounding boxes
                    lines = paragraph.split('\n')
                    filtered_lines = []
                    
                    for line in lines:
                        words = page.extract_words()
                        
                        # Check if any part of the line is inside the table bounding boxes
                        in_table = False
                        for word in words:
                            x0, y0, x1, y1 = word["x0"], word["top"], word["x1"], word["bottom"]
                            # Check if the word's bbox is inside any table bbox
                            for bbox in table_bbox:
                                if x0 >= bbox[0] and x1 <= bbox[2] and y0 >= bbox[1] and y1 <= bbox[3]:
                                    in_table = True
                                    break
                            if in_table:
                                break
                        
                        # If the line is not in a table, add it to the filtered list
                        if not in_table:
                            filtered_lines.append(line)
                    
                    # Combine filtered lines into a single paragraph
                    if filtered_lines:
                        filtered_paragraphs.append(" ".join(filtered_lines))
                
                # Format the paragraphs and add to the final output
                formatted_text = "\n\n".join(filtered_paragraphs)
                all_text.append(f"--- Page {page_num + 1} ---\n{formatted_text}")
        
        # Combine all formatted paragraphs
        complete_text = "\n\n".join(all_text)
        
        # Save to the output file
        with open(output_path, "w", encoding="utf-8") as text_file:
            text_file.write(complete_text)
        
        print(f"Text (excluding tables) extracted and saved to: {output_path}")
    
    except Exception as e:
        print(f"An error occurred: {e}")


# Example Usage
input_pdf = "methodology-sp-us-indices.pdf"  # Replace with your PDF file path
output_txt = "formatted_text_without_tables.txt"  # Replace with your desired text file path
extract_text_without_tables(input_pdf, output_txt)







import fitz  # PyMuPDF
import re
from collections import defaultdict

def extract_participants(pdf_path):
    """ Extracts participant names from the 'Call Participants' section """
    doc = fitz.open(pdf_path)
    participants = set()
    capture = False  # Flag to start capturing names

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            if "Call Participants" in line:
                capture = True  # Start capturing names
            elif capture and line.strip() == "":
                capture = False  # Stop capturing when an empty line is encountered
            elif capture:
                participants.add(line.strip())  # Add name to set
    
    return participants

def extract_text_by_speaker(pdf_path, participants):
    """ Extracts and groups text spoken by each participant """
    doc = fitz.open(pdf_path)
    text_by_speaker = defaultdict(str)
    current_speaker = None

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            if line.strip() in participants:  # If the line is a known speaker
                current_speaker = line.strip()
                continue  # Move to the next line

            if current_speaker:
                text_by_speaker[current_speaker] += line + " "

    # Convert dictionary to a list of dictionaries
    result = [{speaker: text.strip()} for speaker, text in text_by_speaker.items()]
    return result

# Example Usage
pdf_path = "/mnt/data/3Q-24-Earnings-Call-Transcript (1).pdf"

# Step 1: Extract participants
participants = extract_participants(pdf_path)

# Step 2: Extract text and group by speaker
output = extract_text_by_speaker(pdf_path, participants)

# Print output
for entry in output:
    print(entry)
    







import fitz  # PyMuPDF
import json
from collections import defaultdict

def extract_executive_names(pdf_path):
    """ Extracts executive names from the 'Call Participants' section, ignoring roles. """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  # Flag to capture names

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for i in range(len(lines)):
            line = lines[i].strip()

            # Start capturing names after "EXECUTIVES"
            if "EXECUTIVES" in line:
                capture = True
                continue
            
            # Stop capturing when "ANALYSTS" is reached
            if "ANALYSTS" in line:
                capture = False
                break

            # Capture executive names while skipping empty lines and roles
            if capture and line and not any(word in line for word in ["Chief", "Vice", "President", "CEO", "Officer"]):
                executives.append(line)

    return set(executives)  # Return as a unique set

def extract_text_by_executive(pdf_path, executives):
    """ Extracts and groups spoken content by each executive while removing headers/footers. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None

    for page in doc:  # Includes last page now
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Ignore headers/footers
            if "Copyright © 2024 S&P Global" in line or "spglobal.com" in line:
                continue

            # Ignore Operator and Analysts
            if line.lower().startswith(("operator", "analyst", "q&a")):
                current_speaker = None
                continue

            # Detect when an executive starts speaking
            if line in executives:
                current_speaker = line
                continue  # Skip to next line

            # Append content to the respective executive
            if current_speaker:
                text_by_executive[current_speaker] += line + " "

    return text_by_executive

# Example Usage
pdf_path = "3Q-24-Earnings-Call-Transcript.pdf"

# Step 1: Extract executive names
executives = extract_executive_names(pdf_path)
print("Extracted Executives:", executives)  # Debugging step

# Step 2: Extract text and group by executive
output_dict = extract_text_by_executive(pdf_path, executives)

# Step 3: Convert to JSON format
json_output = json.dumps(output_dict, indent=4)

# Save to a JSON file
output_file_path = "executives_earnings_call.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json_file.write(json_output)

# Print JSON output
print(json_output)

# Print path to saved JSON file
print(f"JSON saved to: {output_file_path}")











import fitz  # PyMuPDF
import json
from collections import defaultdict

def extract_executive_names(pdf_path):
    """ Extracts executive names from the 'Call Participants' section, ignoring roles. """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  # Flag to capture names

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for i in range(len(lines)):
            line = lines[i].strip()

            # Start capturing names after "EXECUTIVES"
            if "EXECUTIVES" in line:
                capture = True
                continue
            
            # Stop capturing when "ANALYSTS" is reached
            if "ANALYSTS" in line:
                capture = False
                break

            # Capture executive names while skipping empty lines and roles
            if capture and line and not any(word in line for word in ["Chief", "Vice", "President", "CEO", "Officer"]):
                executives.append(line)

    return set(executives)  # Return as a unique set

def extract_text_by_executive(pdf_path, executives):
    """ Extracts and groups spoken content by each executive while removing headers/footers. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None
    first_page_text = ""  # Store first page's general content

    for page_num, page in enumerate(doc):  # Includes first and last pages now
        # Extract text while **excluding tables and images**
        text = page.get_text("text")  
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Ignore headers/footers
            if "Copyright © 2024 S&P Global" in line or "spglobal.com" in line:
                continue

            # Ignore text that might belong to tables (common in financial reports)
            if line.replace(".", "").replace(",", "").isdigit():  # Detect numeric data
                continue  # Skip financial numbers from tables

            # Ignore Operator and Analysts
            if line.lower().startswith(("operator", "analyst", "q&a")):
                current_speaker = None
                continue

            # Store general text from the first page separately (excluding images/tables)
            if page_num == 0:
                first_page_text += line + " "
                continue  # Skip processing this line further

            # Detect when an executive starts speaking
            if line in executives:
                current_speaker = line
                continue  # Skip to next line

            # Append content to the respective executive
            if current_speaker:
                text_by_executive[current_speaker] += line + " "

    # Include first page content in final JSON
    text_by_executive["General First Page Content"] = first_page_text.strip()

    return text_by_executive

# Example Usage
pdf_path = "/mnt/data/3Q-24-Earnings-Call-Transcript.pdf"

# Step 1: Extract executive names
executives = extract_executive_names(pdf_path)
print("Extracted Executives:", executives)  # Debugging step

# Step 2: Extract text and group by executive
output_dict = extract_text_by_executive(pdf_path, executives)

# Step 3: Convert to JSON format
json_output = json.dumps(output_dict, indent=4)

# Save to a JSON file
output_file_path = "/mnt/data/executives_earnings_call.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json_file.write(json_output)

# Print JSON output
print(json_output)

# Print path to saved JSON file
print(f"JSON saved to: {output_file_path}")










import fitz  # PyMuPDF
import json
from collections import defaultdict

def extract_executive_names(pdf_path):
    """ Extracts executive names from the 'Call Participants' section, ignoring roles. """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  # Flag to capture names

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for i in range(len(lines)):
            line = lines[i].strip()

            # Start capturing names after "EXECUTIVES"
            if "EXECUTIVES" in line:
                capture = True
                continue
            
            # Stop capturing when "ANALYSTS" is reached
            if "ANALYSTS" in line:
                capture = False
                break

            # Capture executive names while skipping empty lines and roles
            if capture and line and not any(word in line for word in ["Chief", "Vice", "President", "CEO", "Officer"]):
                executives.append(line)

    return set(executives)  # Return as a unique set

def clean_text(lines):
    """ Removes headers, footers, and table-related numeric-only lines. """
    cleaned_text = []
    for line in lines:
        line = line.strip()

        # Ignore headers/footers
        if "Copyright © 2024 S&P Global" in line or "spglobal.com" in line:
            continue

        # Ignore text that might belong to tables (common in financial reports)
        if line.replace(".", "").replace(",", "").isdigit():  # Detect numeric data
            continue  # Skip financial numbers from tables

        cleaned_text.append(line)

    return " ".join(cleaned_text)  # Return as a single cleaned string

def extract_text_by_executive(pdf_path, executives):
    """ Extracts first page content first, then groups spoken content by each executive. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None
    first_page_lines = []  # Store first page's general content

    # Extract first page content first
    first_page = doc[0]
    first_page_lines.extend(first_page.get_text("text").split("\n"))
    text_by_executive["General First Page Content"] = clean_text(first_page_lines)

    # Extract executive speech from the rest of the document
    for page_num, page in enumerate(doc):
        if page_num == 0:  
            continue  # Skip first page since it's already extracted

        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Ignore Operator and Analysts
            if line.lower().startswith(("operator", "analyst", "q&a")):
                current_speaker = None
                continue

            # Detect when an executive starts speaking
            if line in executives:
                current_speaker = line
                continue  # Skip to next line

            # Append content to the respective executive
            if current_speaker:
                text_by_executive[current_speaker] += line + " "

    return text_by_executive

# Example Usage
pdf_path = "/mnt/data/3Q-24-Earnings-Call-Transcript.pdf"

# Step 1: Extract executive names
executives = extract_executive_names(pdf_path)
print("Extracted Executives:", executives)  # Debugging step

# Step 2: Extract text and group by executive
output_dict = extract_text_by_executive(pdf_path, executives)

# Step 3: Convert to JSON format
json_output = json.dumps(output_dict, indent=4)

# Save to a JSON file
output_file_path = "/mnt/data/executives_earnings_call.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json_file.write(json_output)

# Print JSON output
print(json_output)

# Print path to saved JSON file
print(f"JSON saved to: {output_file_path}")










import fitz  # PyMuPDF
import json
from collections import defaultdict

# Define bounding box thresholds (adjust if needed)
HEADER_THRESHOLD = 100  # Y-position below which the header ends
FOOTER_THRESHOLD = 700  # Y-position above which the footer starts
TABLE_MIN_HEIGHT = 50  # Minimum height to classify as a table

def extract_executive_names(pdf_path):
    """ Extracts executive names from the 'Call Participants' section, ignoring roles. """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  # Flag to capture names

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for i in range(len(lines)):
            line = lines[i].strip()

            # Start capturing names after "EXECUTIVES"
            if "EXECUTIVES" in line:
                capture = True
                continue
            
            # Stop capturing when "ANALYSTS" is reached
            if "ANALYSTS" in line:
                capture = False
                break

            # Capture executive names while skipping empty lines and roles
            if capture and line and not any(word in line for word in ["Chief", "Vice", "President", "CEO", "Officer"]):
                executives.append(line)

    return set(executives)  # Return as a unique set

def clean_text(lines):
    """ Removes table-related numeric-only lines. """
    cleaned_text = []
    for line in lines:
        line = line.strip()

        # Ignore text that might belong to tables (common in financial reports)
        if line.replace(".", "").replace(",", "").isdigit():  # Detect numeric data
            continue  # Skip financial numbers from tables

        cleaned_text.append(line)

    return " ".join(cleaned_text)  # Return as a single cleaned string

def extract_text_by_executive(pdf_path, executives):
    """ Extracts first page content first, then groups spoken content by each executive. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None
    first_page_lines = []  # Store first page's general content

    def extract_clean_text_from_page(page):
        """ Extracts text while removing headers, footers, and tables/images. """
        blocks = page.get_text("blocks")  # Get text blocks with bounding box data
        extracted_text = []

        for b in blocks:
            x0, y0, x1, y1, text = b  # Bounding box and text

            # Remove header & footer based on bounding box position
            if y0 < HEADER_THRESHOLD or y1 > FOOTER_THRESHOLD:
                continue  # Skip header/footer

            # Skip table/image-related text (bounding box with large height)
            if (y1 - y0) > TABLE_MIN_HEIGHT:
                continue

            extracted_text.append(text.strip())

        return clean_text(extracted_text)

    # Extract first page content first (removing header/footer/tables/images)
    first_page = doc[0]
    first_page_text = extract_clean_text_from_page(first_page)
    text_by_executive["General First Page Content"] = first_page_text

    # Extract executive speech from the rest of the document
    for page_num, page in enumerate(doc):
        if page_num == 0:  
            continue  # Skip first page since it's already extracted

        text = extract_clean_text_from_page(page)
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Ignore Operator and Analysts
            if line.lower().startswith(("operator", "analyst", "q&a")):
                current_speaker = None
                continue

            # Detect when an executive starts speaking
            if line in executives:
                current_speaker = line
                continue  # Skip to next line

            # Append content to the respective executive
            if current_speaker:
                text_by_executive[current_speaker] += line + " "

    return text_by_executive

# Example Usage
pdf_path = "/mnt/data/3Q-24-Earnings-Call-Transcript.pdf"

# Step 1: Extract executive names
executives = extract_executive_names(pdf_path)
print("Extracted Executives:", executives)  # Debugging step

# Step 2: Extract text and group by executive
output_dict = extract_text_by_executive(pdf_path, executives)

# Step 3: Convert to JSON format
json_output = json.dumps(output_dict, indent=4)

# Save to a JSON file
output_file_path = "/mnt/data/executives_earnings_call.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json_file.write(json_output)

# Print JSON output
print(json_output)

# Print path to saved JSON file
print(f"JSON saved to: {output_file_path}")









import fitz  # PyMuPDF
import json
from collections import defaultdict

# Define bounding box thresholds (adjust if needed)
HEADER_THRESHOLD = 100  # Y-position below which the header ends
FOOTER_THRESHOLD = 700  # Y-position above which the footer starts
TABLE_MIN_HEIGHT = 50  # Minimum height to classify as a table

def extract_executive_names(pdf_path):
    """ Extracts executive names from the 'Call Participants' section, ignoring roles. """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  # Flag to capture names

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for i in range(len(lines)):
            line = lines[i].strip()

            # Start capturing names after "EXECUTIVES"
            if "EXECUTIVES" in line:
                capture = True
                continue
            
            # Stop capturing when "ANALYSTS" is reached
            if "ANALYSTS" in line:
                capture = False
                break

            # Capture executive names while skipping empty lines and roles
            if capture and line and not any(word in line for word in ["Chief", "Vice", "President", "CEO", "Officer"]):
                executives.append(line)

    return set(executives)  # Return as a unique set

def clean_text(lines):
    """ Removes table-related numeric-only lines. """
    cleaned_text = []
    for line in lines:
        line = line.strip()

        # Ignore text that might belong to tables (common in financial reports)
        if line.replace(".", "").replace(",", "").isdigit():  # Detect numeric data
            continue  # Skip financial numbers from tables

        cleaned_text.append(line)

    return " ".join(cleaned_text)  # Return as a single cleaned string

def extract_text_by_executive(pdf_path, executives):
    """ Extracts first page content first, then groups spoken content by each executive. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None
    first_page_lines = []  # Store first page's general content

    def extract_clean_text_from_page(page):
        """ Extracts text while removing headers, footers, and tables/images. """
        blocks = page.get_text("blocks")  # Get text blocks with bounding box data
        extracted_text = []

        for b in blocks:
            x0, y0, x1, y1, text, *_ = b  # Ignore extra values beyond the first five

            # Remove header & footer based on bounding box position
            if y0 < HEADER_THRESHOLD or y1 > FOOTER_THRESHOLD:
                continue  # Skip header/footer

            # Skip table/image-related text (bounding box with large height)
            if (y1 - y0) > TABLE_MIN_HEIGHT:
                continue

            extracted_text.append(text.strip())

        return clean_text(extracted_text)

    # Extract first page content first (removing header/footer/tables/images)
    first_page = doc[0]
    first_page_text = extract_clean_text_from_page(first_page)
    text_by_executive["General First Page Content"] = first_page_text

    # Extract executive speech from the rest of the document
    for page_num, page in enumerate(doc):
        if page_num == 0:  
            continue  # Skip first page since it's already extracted

        text = extract_clean_text_from_page(page)
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Ignore Operator and Analysts
            if line.lower().startswith(("operator", "analyst", "q&a")):
                current_speaker = None
                continue

            # Detect when an executive starts speaking
            if line in executives:
                current_speaker = line
                continue  # Skip to next line

            # Append content to the respective executive
            if current_speaker:
                text_by_executive[current_speaker] += line + " "

    return text_by_executive

# Example Usage
pdf_path = "/mnt/data/3Q-24-Earnings-Call-Transcript.pdf"

# Step 1: Extract executive names
executives = extract_executive_names(pdf_path)
print("Extracted Executives:", executives)  # Debugging step

# Step 2: Extract text and group by executive
output_dict = extract_text_by_executive(pdf_path, executives)

# Step 3: Convert to JSON format
json_output = json.dumps(output_dict, indent=4)

# Save to a JSON file
output_file_path = "/mnt/data/executives_earnings_call.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json_file.write(json_output)

# Print JSON output
print(json_output)

# Print path to saved JSON file
print(f"JSON saved to: {output_file_path}")

