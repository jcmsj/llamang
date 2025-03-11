import os
import fitz  # PyMuPDF
import json
import argparse
import glob
from pytesseract import image_to_boxes

def extract_pdf_form_fields(pdf_path):
    """Extract form fields from a PDF file."""
    form_fields = {}
    
    # Open the PDF file
    doc = fitz.open(pdf_path)
    
    # Get form fields
    for page_num in range(len(doc)):
        page = doc[page_num]
        form_fields[f"page_{page_num+1}"] = []
        
        # Get form fields on the current page
        widgets = page.widgets()
        for widget in widgets:
            field_info = {
                "name": widget.field_name,
                "type": widget.field_type_string,
                "value": widget.field_value,
                "rect": [float(num) for num in widget.rect],  # Convert rect to list of floats
                # "flags": widget.field_flags,
            }
            
            # Add additional properties based on field type
            if widget.field_type_string == "Text":
                field_info["text_format"] = widget.text_format
            elif widget.field_type_string == "Choice":
                field_info["choice_options"] = widget.choice_options
            
            form_fields[f"page_{page_num+1}"].append(field_info)
    
    # Close the document
    doc.close()
    
    return form_fields

def to_csv(form_fields, output_csv):
    """Write form fields to a CSV file."""
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("Page,Field Name,Field Type,Field Value,Field Rect\n")
        for page_num, fields in form_fields.items():
            for field in fields:
                f.write(f"{page_num},{field['name']},{field['type']},{field['value']},\"{field['rect']}\"\n")


def process_pdf(input_pdf, output=None, output_format='csv'):
    """Process a single PDF file."""
    # Set default output file using basename of input file
    if output is None:
        output = f"{os.path.splitext(os.path.basename(input_pdf))[0]}.{output_format}"
    
    try:
        # Extract form fields
        form_fields = extract_pdf_form_fields(input_pdf)
        
        # Write output
        if output_format == 'csv':
            to_csv(form_fields, output)
        else:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(form_fields, f, indent=4)
        
        print(f"Form fields from {input_pdf} successfully extracted to {output}")
        return True
    
    except Exception as e:
        print(f"Error processing {input_pdf}: {e}")
        return False


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Extract form fields from PDF files into JSON or CSV files")
    parser.add_argument("input_path", help="Path to the input PDF file or folder containing PDF files")
    parser.add_argument("-o", "--output", help="Path to the output file (default: based on input filename)")
    parser.add_argument("-f", "--format", default='csv', choices=['csv', 'json'], 
                        help="Output format (json or csv)")
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input path is a directory
    if os.path.isdir(args.input_path):
        # Find all PDF files in the directory
        pdf_files = glob.glob(os.path.join(args.input_path, "*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {args.input_path}")
            return
            
        # Process each PDF file
        success_count = 0
        for pdf_file in pdf_files:
            output_file = None
            if args.output:
                # If output is specified and it's a directory, use it as the output directory
                if os.path.isdir(args.output):
                    base_name = f"{os.path.splitext(os.path.basename(pdf_file))[0]}.{args.format}"
                    output_file = os.path.join(args.output, base_name)
                else:
                    # If single output file is specified for multiple PDFs, skip
                    print("Warning: Single output file specified for multiple PDFs. Using default naming.")
            
            # Process the PDF file
            if process_pdf(pdf_file, output_file, args.format):
                success_count += 1
        
        print(f"Processed {success_count} out of {len(pdf_files)} PDF files")
    
    else:
        # Process single PDF file
        process_pdf(args.input_path, args.output, args.format)


if __name__ == "__main__":
    main()
