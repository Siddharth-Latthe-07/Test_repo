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
    Extracts all text from a PDF file, excluding text inside tables.
    
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
                
                # Extract raw text with bounding boxes
                text_objects = page.extract_words()
                
                # Filter out text inside table bounding boxes
                filtered_text = []
                for obj in text_objects:
                    x0, y0, x1, y1 = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                    in_table = any(
                        x0 >= bbox[0] and x1 <= bbox[2] and y0 >= bbox[1] and y1 <= bbox[3]
                        for bbox in table_bbox
                    )
                    if not in_table:
                        filtered_text.append(obj["text"])
                
                # Format the text and append to the final output
                formatted_text = format_text(" ".join(filtered_text))
                all_text.append(f"--- Page {page_num + 1} ---\n{formatted_text}")
        
        # Combine all formatted text
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









import fitz

def extract_text_without_tables_images(pdf_path):
    doc = fitz.open(pdf_path)
    extracted_text = ""

    for page in doc:
        blocks = page.get_text("dict")["blocks"]  # Extract text as blocks

        for block in blocks:
            if "lines" in block:  # Ignore images and tables
                for line in block["lines"]:
                    for span in line["spans"]:
                        font = span["font"]
                        text = span["text"]

                        # Check for bold or italic fonts
                        is_bold = "Bold" in font or "Black" in font
                        is_italic = "Italic" in font or "Oblique" in font

                        if is_bold:
                            extracted_text += f"**{text}** "  # Markdown bold
                        elif is_italic:
                            extracted_text += f"*{text}* "  # Markdown italic
                        else:
                            extracted_text += f"{text} "
                    
                    extracted_text += "\n"  # Add new line after each line in a block

        extracted_text += "\n"  # New paragraph after each block

    return extracted_text.strip()

# Example usage
pdf_path = "sample.pdf"  # Replace with your file path
text = extract_text_without_tables_images(pdf_path)
print(text)
