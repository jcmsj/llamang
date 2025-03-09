import json
import fitz  # PyMuPDF
import csv
import ollama
import argparse
import os
from datetime import datetime

from maker.ext import extract_pdf_form_fields, to_csv

def read_form_fields_from_csv(csv_path):
    """Read form field information from a CSV file."""
    fields_by_page = {}
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            page = row['Page']
            if page not in fields_by_page:
                fields_by_page[page] = []
            
            # Create a field dictionary without the rect information
            field = {
                'name': row['Field Name'],
                'type': row['Field Type'],
                'value': row['Field Value']
            }
            
            # Keep the rect information separately for later use
            field['rect'] = row.get('Field Rect', None)
            
            fields_by_page[page].append(field)
    
    return fields_by_page

def generate_form_values(fields, model_name):
    """Generate new values for form fields using Ollama."""
    # Group fields by type
    text_fields = [field for field in fields if field['type'] == 'Text']
    # we will only modify text fields, other fields such as Buttons, Checkboxes, etc. will be left as is
    # other_fields = [field for field in fields if field['type'] != 'Text']
    keyval = {}
    for field in fields:
        keyval[field['name']] = field['value']
    # Only process text fields with Ollama
    if not text_fields:
        return fields
    
    # Construct prompt
    prompt = "Generate realistic but fictional values for the following form fields based on this json information\n\n"

    prompt += f"<Form Fields>\n{keyval}\n</Form Fields>\n\n"

    # Final output as json
    steps = [
        "Please generate appropriate data for each field, maintaining coherence between related fields.",
        "Try to infer based on the field name what format the data should be in.",
        "Consider also based on the field name when it's ok to reuse a previous value.",
        "No need to provide explanations or context.",
        "output as a JSON object with field names as keys and your generated value as string."
    ]
    prompt += "Instructions:\n"
    for i in range(len(steps)):
        prompt += f"{i+1}. {steps[i]}\n"

    # # save prompt to a file
    # with open('prompt.txt', 'w') as f:
    #     f.write(prompt)
    # # Call Ollama model
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            format='json'
        )
        generated_text = response['message']['content'].strip()
        
        # Parse the JSON response
        results = json.loads(generated_text)

        # Update the text fields with the generated values
        for field in text_fields:
            field['value'] = results.get(field['name'], field['value'])
        
        return text_fields    
    except Exception as e:
        print(f"Error generating values: {e}")
        return fields


def update_pdf_form(pdf_path, fields_by_page, output_path):
    """Update the PDF form with the generated values."""
    doc = fitz.open(pdf_path)
    
    for page_key, fields in fields_by_page.items():
        # Extract page number from the key (e.g., "page_1" -> 0)
        page_num = int(page_key.split('_')[1]) - 1
        
        if page_num < len(doc):
            page = doc[page_num]
            
            # Update each field on this page
            for field in fields:
                if field['name']:  # Only process fields with names
                    widgets = page.widgets()
                    for widget in widgets:
                        if widget.field_name == field['name']:
                            widget.field_value = field['value']
                            widget.update()
    
    # Save the updated PDF
    doc.save(output_path)
    doc.close()


def save_generated_fields_to_csv(fields_by_page, output_csv):
    """Save the generated field values to a CSV file."""
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Page', 'Field Name', 'Field Type', 'Field Value'])
        
        for page_key, fields in fields_by_page.items():
            for field in fields:
                writer.writerow([
                    page_key,
                    field['name'],
                    field['type'],
                    field['value']
                ])


def main():
    parser = argparse.ArgumentParser(description="Generate form field values and update PDF forms")
    parser.add_argument("input_pdf", help="Path to the input PDF template")
    parser.add_argument("input_csv", help="Path to the CSV with form field information")
    parser.add_argument("-o", "--output", help="Path to the output PDF file")
    parser.add_argument("-c", "--csv", help="Path to the output CSV file for generated values")
    parser.add_argument("-m", "--model", default="llama3.2:latest", help="Ollama model to use (default: llama3.2:latest)")
    
    args = parser.parse_args()
    
    # Generate default output paths if not provided
    if not args.output:
        base_name = os.path.splitext(os.path.basename(args.input_pdf))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"{base_name}_generated_{timestamp}.pdf"
    
    if not args.csv:
        base_name = os.path.splitext(os.path.basename(args.input_csv))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.csv = f"{base_name}_generated_{timestamp}.csv"
    
    try:
        # Read form fields from CSV
        print(f"Reading form fields from {args.input_csv}")
        fields_by_page = read_form_fields_from_csv(args.input_csv)
        
        # Generate new values page by page
        print(f"Generating new values with {args.model} model")
        updated_fields_by_page = {}
        for page_key, fields in fields_by_page.items():
            print(f"Processing {page_key}")
            updated_fields = generate_form_values(fields, args.model)
            updated_fields_by_page[page_key] = updated_fields
        
        # Update the PDF form
        print(f"Updating PDF form and saving to {args.output}")
        update_pdf_form(args.input_pdf, updated_fields_by_page, args.output)
        
        # Reuse the updated fields to save to CSV
        generated_fields = extract_pdf_form_fields(args.output)
        to_csv(generated_fields, args.csv)
        
        print("Done!")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {e}" )


if __name__ == "__main__":
    main()
