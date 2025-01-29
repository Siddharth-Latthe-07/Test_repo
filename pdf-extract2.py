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
