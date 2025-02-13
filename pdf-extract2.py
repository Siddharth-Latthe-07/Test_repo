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







import fitz  # PyMuPDF
import json
from collections import defaultdict
import re

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

def clean_text(text):
    """ Removes headers, footers, and unwanted repeated phrases from extracted text. """
    # Patterns to remove
    unwanted_patterns = [
        r"Copyright © \d{4} S&P Global.*?",  # S&P Global copyright notice
        r"spglobal\.com.*?",  # Any URLs related to S&P Global
        r"ICL GROUP LTD .*?EARNINGS CALL.*?",  # Earnings call title
        r"Page \d+",  # Page numbers if present
        r"www\..*?",  # Any general website links
        r"https?://[^\s]+",  # Any hyperlinks
    ]

    # Apply regex patterns to remove unwanted text
    for pattern in unwanted_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return text.strip()  # Return cleaned text without extra spaces

def extract_text_by_executive(pdf_path, executives):
    """ Extracts and groups spoken content by each executive while removing headers/footers. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None

    for page in doc:
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

    # Clean extracted text
    for speaker in text_by_executive:
        text_by_executive[speaker] = clean_text(text_by_executive[speaker])

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

def remove_headers_footers(lines):
    """ Removes headers and footers from a page's text. """
    cleaned_lines = []
    for line in lines:
        # Skip headers/footers with specific patterns
        if any(keyword in line.lower() for keyword in [
            "copyright", "s&p global", "market intelligence", "earnings call", "conference call", "call transcript"
        ]):
            continue
        cleaned_lines.append(line.strip())
    
    return cleaned_lines

def extract_text_by_executive(pdf_path, executives):
    """ Extracts and groups spoken content by each executive while removing headers/footers. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None

    for page in doc:  # Includes last page now
        text = page.get_text("text")
        lines = text.split("\n")

        # Remove headers and footers before processing
        lines = remove_headers_footers(lines)

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
import re
from collections import defaultdict

def extract_executive_names(pdf_path):
    """ Extracts executive names from the 'Call Participants' section robustly. """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  # Flag to capture names
    stop_keywords = ["analysts", "moderator", "operator", "questions and answers"]  # Stop capturing on these
    role_keywords = ["chief", "vice", "president", "ceo", "officer", "head"]  # Role-based filtering

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for i in range(len(lines)):
            line = lines[i].strip().lower()

            # Start capturing after "EXECUTIVES" or "Call Participants"
            if "executives" in line or "call participants" in line:
                capture = True
                continue
            
            # Stop capturing if a stopping keyword appears
            if any(keyword in line for keyword in stop_keywords):
                capture = False
                break  

            # Capture executive names, but skip role descriptions
            if capture and line and not any(word in line for word in role_keywords):
                executives.append(lines[i].strip())  # Store original case name

    return set(executives)  # Return unique executive names

def remove_headers_footers(lines):
    """ Removes headers and footers from a page's text. """
    cleaned_lines = []
    for line in lines:
        # Skip headers/footers with specific patterns
        if any(keyword in line.lower() for keyword in [
            "copyright", "s&p global", "market intelligence", "earnings call", "conference call", "call transcript"
        ]):
            continue
        cleaned_lines.append(line.strip())
    
    return cleaned_lines

def extract_text_by_executive(pdf_path, executives):
    """ Extracts and groups spoken content by each executive while removing headers/footers. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None

    for page in doc:  # Includes last page now
        text = page.get_text("text")
        lines = text.split("\n")

        # Remove headers and footers before processing
        lines = remove_headers_footers(lines)

        for line in lines:
            line = line.strip()

            # Ignore Operator and Analysts
            if line.lower().startswith(("operator", "analyst", "q&a", "moderator")):
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

# Step 1: Extract executive names (robust method)
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
import re

def extract_executive_names(pdf_path):
    """ Extracts executive names from the 'Call Participants' section, stopping at any full-uppercase heading. """
    doc = fitz.open(pdf_path)
    executives = []
    capture = False  # Flag to capture names

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Start capturing after "EXECUTIVES"
            if "EXECUTIVES" in line:
                capture = True
                continue

            # Stop capturing at any full uppercase word (new section)
            if capture and line.isupper():
                capture = False
                break

            # Capture only names (excluding titles like CFO, VP, etc.)
            if capture and line and not any(word in line for word in ["Chief", "Vice", "President", "CEO", "Officer", "Relations","CFO"]):
                executives.append(line)

    return set(executives)  # Return unique set of names

def extract_text_by_executive(pdf_path, executives):
    """ Extracts and groups spoken content by each executive while removing headers/footers. """
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Ignore headers/footers
            if "Copyright ©" in line or "spglobal.com" in line:
                continue

            # Ignore Operator and Analysts
            if re.match(r"^(Operator|Analyst|Q&A|Question and Answer|Moderator)", line, re.IGNORECASE):
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
pdf_path = "q2-2024-transcript.pdf"

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

def detect_executive_style(pdf_path):
    """Detects the text style (font, size) of 'EXECUTIVES' to dynamically extract executives."""
    doc = fitz.open(pdf_path)

    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]  # Extract text blocks
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if "EXECUTIVES" in span["text"]:
                            return span["font"], span["size"]  # Return font and size of "EXECUTIVES"

    return None, None  # Return None if not found

def extract_executive_names(pdf_path, exec_font, exec_size):
    """Extracts executive names using the detected style."""
    doc = fitz.open(pdf_path)
    executives = []
    capture = False

    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        # Start capturing after 'EXECUTIVES' detected
                        if text == "EXECUTIVES" and span["font"] == exec_font and span["size"] == exec_size:
                            capture = True
                            continue

                        # Stop capturing when another text of the same style appears
                        if capture and text.isupper() and span["font"] == exec_font and span["size"] == exec_size:
                            capture = False
                            break

                        # Capture executive names (avoid roles)
                        if capture and text and not any(word in text for word in ["Chief", "Vice", "President", "CEO", "Officer", "Relations"]):
                            executives.append(text)

    return set(executives)

def extract_executive_speeches(pdf_path, executives):
    """Extracts and groups spoken content by each executive."""
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None

    for page in doc:
        text = page.get_text("text")
        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Ignore headers/footers
            if "Copyright ©" in line or "spglobal.com" in line:
                continue

            # Ignore Operator and Analysts
            if line.lower().startswith(("operator", "analyst", "q&a", "stakeholders", "question and answer")):
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
pdf_path = "/mnt/data/q2-2024-transcript_1.pdf"

# Step 1: Detect style of "EXECUTIVES"
exec_font, exec_size = detect_executive_style(pdf_path)
if not exec_font or not exec_size:
    print("Error: 'EXECUTIVES' style not found.")
    exit()

# Step 2: Extract executive names using the detected style
executives = extract_executive_names(pdf_path, exec_font, exec_size)
print("Extracted Executives:", executives)

# Step 3: Extract speeches of executives
output_dict = extract_executive_speeches(pdf_path, executives)

# Step 4: Convert to JSON format
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

def detect_executive_style(pdf_path):
    """Detects the text style (font, size) of 'EXECUTIVES' dynamically."""
    doc = fitz.open(pdf_path)

    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]  
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        if "EXECUTIVES" in span["text"]:
                            return span["font"], span["size"]  # Return font and size of "EXECUTIVES"

    return None, None  # Return None if not found

def extract_executive_names(pdf_path, exec_font, exec_size):
    """Extracts only executive names using the detected text style."""
    doc = fitz.open(pdf_path)
    executives = []
    capture = False

    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        # Start capturing after 'EXECUTIVES'
                        if text == "EXECUTIVES" and span["font"] == exec_font and span["size"] == exec_size:
                            capture = True
                            continue

                        # Stop capturing when another text of the same style appears (new section)
                        if capture and text.isupper() and span["font"] == exec_font and span["size"] == exec_size:
                            capture = False
                            break

                        # Capture only executive names, ignore roles
                        if capture and text and not any(word in text for word in ["Chief", "Vice", "President", "CEO", "Officer", "Relations"]):
                            executives.append(text)

    return set(executives)

def extract_executive_speeches(pdf_path, executives):
    """Extracts and groups only executives' spoken content throughout the document."""
    doc = fitz.open(pdf_path)
    text_by_executive = defaultdict(str)
    current_speaker = None

    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        # Detect an executive's speech
                        if text in executives:
                            current_speaker = text
                            continue  # Move to the next line

                        # Capture text spoken only by executives
                        if current_speaker:
                            text_by_executive[current_speaker] += text + " "

    return text_by_executive

# Example Usage
pdf_path = "/mnt/data/q2-2024-transcript_1.pdf"

# Step 1: Detect style of "EXECUTIVES"
exec_font, exec_size = detect_executive_style(pdf_path)
if not exec_font or not exec_size:
    print("Error: 'EXECUTIVES' style not found.")
    exit()

# Step 2: Extract executive names
executives = extract_executive_names(pdf_path, exec_font, exec_size)
print("Extracted Executives:", executives)  # Debugging

# Step 3: Extract only executives' spoken text
executive_speeches = extract_executive_speeches(pdf_path, executives)

# Step 4: Convert to JSON format
json_output = json.dumps(executive_speeches, indent=4)

# Save to JSON file
output_file_path = "executives_speeches.json"
with open(output_file_path, "w", encoding="utf-8") as json_file:
    json_file.write(json_output)

# Print JSON output
print(json_output)

# Print path to saved JSON file
print(f"JSON saved to: {output_file_path}")












import fitz  # PyMuPDF
import json
import os
import pandas as pd
from collections import defaultdict

# Initialize variables
full_text = ""
all_tables = []
current_name = None
current_briefing = []
data = {}

HEADER_HEIGHT = 50  # Adjust based on your document
FOOTER_HEIGHT = 50  # Adjust based on your document

input_folder = "input2/01-01-2023 to 28-02-2023"  # Folder containing multiple PDFs
output_folder = "output"
os.makedirs(output_folder, exist_ok=True)

# Function to read roles from a text file
def read_roles(file_path):
    with open(file_path, "r") as file:
        roles = [line.strip() for line in file]
    return roles

# Read roles from the 'roles.txt' file
roles_file_path = "roles.txt"
executive_roles = read_roles(roles_file_path)

def add_to_data(current_name, current_briefing, new_name=None):
    """Adds data to dictionary ensuring names are mapped to spoken text correctly."""
    global data
    if current_name and current_briefing:
        if current_name in data:
            data[current_name] += " ".join(current_briefing)
        else:
            data[current_name] = " ".join(current_briefing)
    return new_name, []

def extract_executive_names(pdf_path):
    """Extracts executive names from the 'Call Participants' section, combining multi-line names."""
    doc = fitz.open(pdf_path)
    executives = []
    
    for page in doc:
        text_blocks = page.get_text("dict")["blocks"]
        capture = False  # Flag to start capturing executive names
        previous_font = None
        name_buffer = []

        for block in text_blocks:
            if "lines" not in block:
                continue

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()

                    # Start capturing names if "EXECUTIVES" is found
                    if "EXECUTIVES" in text:
                        capture = True
                        continue

                    if capture:
                        font = span["font"]

                        # If font is same as previous, append to name buffer
                        if previous_font == font:
                            name_buffer.append(text)
                        else:
                            if name_buffer:
                                full_name = " ".join(name_buffer)
                                if full_name.isupper():
                                    executives.append(full_name)
                            name_buffer = [text]  # Start new name group

                        previous_font = font  # Store font for next iteration
        
        # Capture last name buffer
        if name_buffer:
            full_name = " ".join(name_buffer)
            if full_name.isupper():
                executives.append(full_name)

    return set(executives)

def process_page(page, page_num, current_name, current_briefing, doc_name):
    """Processes a single page, extracting tables and spoken text."""
    global data
    blocks = page.get_text("dict")["blocks"]
    table_bboxes = []

    # Extract tables
    tables = page.find_tables()
    for table in tables:
        is_highlighted = any(
            fitz.Rect(table.bbox).intersects(fitz.Rect(annot.rect)) for annot in page.annots() if annot
        )
        if not is_highlighted:
            all_tables.append({"document": doc_name, "page": page_num, "data": table.extract()})
        table_bboxes.append(table.bbox)

    # Extract text
    for block in blocks:
        block_bbox = fitz.Rect(block["bbox"])

        # Ignore text inside tables
        if any(block_bbox.intersects(fitz.Rect(bbox)) for bbox in table_bboxes) or block_bbox.y0 < HEADER_HEIGHT:
            continue

        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if text:
                        is_bold = "Bold" in span["font"]

                        if is_bold:  # Bold text
                            current_name, current_briefing = add_to_data(current_name, current_briefing, text)
                        else:
                            current_briefing.append(text)

    return current_name, current_briefing

def process_pdfs(input_folder, output_folder):
    """Processes all PDFs in the input folder and extracts relevant information."""
    global current_name, current_briefing, data, all_tables
    all_executive_data = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".pdf"):
            doc_path = os.path.join(input_folder, filename)
            doc_name = os.path.splitext(filename)[0]

            # Extract company name and publication date from the file name
            parts = doc_name.split("_")
            company_name = parts[0] if len(parts) > 0 else "Unknown"
            pub_date = parts[2] if len(parts) > 2 else "Unknown"

            doc = fitz.open(doc_path)
            executive_names = extract_executive_names(doc_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                current_name, current_briefing = process_page(page, page_num, current_name, current_briefing, doc_name)

            # Finalize last extracted text
            current_name, current_briefing = add_to_data(current_name, current_briefing)

            # Filter data to keep only text spoken by the extracted executive names
            filtered_data = {name: content for name, content in data.items() if name in executive_names}

            formatted_data = [
                {"executive": name, "briefing": content, "company_name": company_name, "pub_date": pub_date}
                for name, content in filtered_data.items()
            ]
            all_executive_data.extend(formatted_data)

            # Write the filtered data to a file
            with open(os.path.join(output_folder, f"{doc_name}_executive_data.json"), "w") as f:
                json.dump(formatted_data, f, indent=4)

            # Reset variables for the next document
            data = {}

    return all_executive_data

# Process PDFs in the input folder
all_executive_data = process_pdfs(input_folder, output_folder)

# Create a DataFrame from the extracted data
df = pd.DataFrame(all_executive_data, columns=["executive", "briefing", "company_name", "pub_date"])

# Write the DataFrame to an Excel file
excel_output_path = os.path.join(output_folder, "all_executive_data.xlsx")
df.to_excel(excel_output_path, index=False)

print(f"Data saved successfully in {excel_output_path}")






import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def clean_text(text):
    # Merge lines intelligently: assuming names don't end with punctuation
    lines = text.split("\n")
    merged_lines = []
    buffer = ""

    for line in lines:
        stripped_line = line.strip()
        if buffer:  # If there's a buffered line
            buffer += " " + stripped_line  # Merge with next line
            merged_lines.append(buffer)
            buffer = ""  # Reset buffer
        elif stripped_line:  # If the line isn't empty
            buffer = stripped_line
        else:
            if buffer:
                merged_lines.append(buffer)
                buffer = ""

    if buffer:  # Append last buffer if not empty
        merged_lines.append(buffer)

    return "\n".join(merged_lines)

# Example usage
pdf_path = "example.pdf"
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(raw_text)
print(cleaned_text)









import fitz  # PyMuPDF
import json
import os

def extract_executive_names(pdf_path, executive_roles):
    """Extracts executive names from a PDF file."""
    doc = fitz.open(pdf_path)
    executives = []
    capture = False
    previous_bbox = None

    for page in doc:
        blocks = page.get_text("dict")["blocks"]

        for block in blocks:
            if 'lines' in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        bbox = fitz.Rect(span["bbox"])
                        font = span["font"]

                        if text == "EXECUTIVES":
                            capture = True
                            continue

                        if capture:
                            if text.isupper():
                                return set(executives)  # Stop if another section starts

                            if "Italic" not in font:
                                if previous_bbox:
                                    vertical_distance = bbox.y0 - previous_bbox.y1
                                    if vertical_distance < 10:
                                        executives[-1] = executives[-1] + " " + text
                                    else:
                                        executives.append(text)
                                else:
                                    executives.append(text)
                                
                                previous_bbox = bbox

    cleaned_executives = set()
    for name in executives:
        cleaned_name = " ".join(word for word in name.split() if word not in executive_roles)
        cleaned_executives.add(cleaned_name)

    return cleaned_executives


def extract_executive_dialogues(pdf_path, executive_names):
    """Extracts text spoken by executives from a PDF file."""
    doc = fitz.open(pdf_path)
    executive_texts = {name: [] for name in executive_names}
    capture = None  # Store the active speaker
    last_page_index = len(doc) - 1

    for page_num, page in enumerate(doc):
        if page_num == last_page_index:  # Ignore the last page
            continue

        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if 'lines' in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()

                        if text in executive_names:  
                            capture = text  # Set the active speaker
                        elif capture:  
                            executive_texts[capture].append(text)

    # Convert list of texts into concatenated string for each executive
    for name in executive_texts:
        executive_texts[name] = " ".join(executive_texts[name])

    return executive_texts


def process_pdfs(pdf_folder, executive_roles, output_json):
    """Processes all PDFs in a folder and extracts executive dialogues."""
    all_executive_data = {}

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)

            # Extract executive names
            executive_names = extract_executive_names(pdf_path, executive_roles)
            if not executive_names:
                continue  # Skip if no executives found

            # Extract spoken text by executives
            executive_dialogues = extract_executive_dialogues(pdf_path, executive_names)

            # Merge data into main dictionary
            for name, text in executive_dialogues.items():
                if name in all_executive_data:
                    all_executive_data[name] += " " + text
                else:
                    all_executive_data[name] = text

    # Save as JSON
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(all_executive_data, json_file, indent=4, ensure_ascii=False)

    print(f"Extraction completed! Data saved in {output_json}")


# Example Usage
pdf_folder_path = "path_to_your_pdf_directory"
executive_roles_set = {"CEO", "CFO", "COO", "President", "Vice", "Manager"}  # Define roles to exclude from names
output_json_path = "executive_speeches.json"

process_pdfs(pdf_folder_path, executive_roles_set, output_json_path)
