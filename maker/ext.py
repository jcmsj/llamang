import fitz  # PyMuPDF
import json
import argparse


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

def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Extract form fields from a PDF file into a JSON file")
    parser.add_argument("input_pdf", help="Path to the input PDF file")
    parser.add_argument("-o", "--output", help="Path to the output JSON file (default: fields.json)", default="fields.json")
    # --format, -f: json|csv
    parser.add_argument("-f", "--format", default='csv', help="Output format (json or csv)")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Extract form fields
        form_fields = extract_pdf_form_fields(args.input_pdf)
        
        # Write to JSON file
        if args.format == 'csv':
            # Write to CSV file
            to_csv(form_fields, args.output)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(form_fields, f, indent=4)
        
        print(f"Form fields successfully extracted to {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
