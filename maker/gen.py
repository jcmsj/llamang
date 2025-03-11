import copy
import json
from random import randint, random
import fillpdf
import fillpdf.fillpdfs
import ollama
import argparse
import os
from datetime import datetime
from faker import Faker
from fields import read_form_fields_from_csv
from ocr import (
    OCRException,
    fits_to_imgs,
    to_dict,
    update_rects_to_actual_textbox,
)
from update_pdf_form import update_pdf_form

def generate_hash(previous_hash=None):
    """Generate a new hash or derive one from a previous hash for variation randomization."""
    if previous_hash is None:
        # Generate a completely new random hash
        return hash(str(random()) + str(datetime.now()))
    else:
        # Derive a new hash from the previous one
        return hash(str(previous_hash) + str(random()))

def generate_form_values(fields, model_name, document_title, fallback_model=None, min_fill_percentage=0.6):
    """Generate new values for form fields using Ollama."""
    # Group fields by type
    text_fields = [field for field in fields if field["type"] == "Text"]
    # we will only modify text fields, other fields such as Buttons, Checkboxes, etc. will be left as is
    # other_fields = [field for field in fields if field['type'] != 'Text']
    keyval = {}
    for field in fields:
        keyval[field["name"]] = field["value"]
    # Only process text fields with Ollama
    if not text_fields:
        return fields

    faker = Faker()

    # Construct prompt
    prompt = ""

    # Final output as json
    steps = [
        f"ROLE: You're normally a `{faker.job()}`, but today you're an assistant at a Patent Office",
        "Your task is to make random data for each field, maintaining coherence between the given fields",
        "Infer field value based on field name",
        "you may reuse values based on field name",
        "don't use unusual characters for values",
        f"Be very creative for named entities",
        "field values should not be nested json",
        "Properly escape commas that would disrupt the JSON format",
        "No need to provide explanations or context",
        "Output as a JSON object with field names as keys and your generated value as string"
    ]

    prompt += "Instructions:\n"
    for i in range(len(steps)):
        prompt += f"{i+1}. {steps[i]}\n"

    # Add the document title to the prompt
    prompt += f"\nhere's a hint about the information source (don't use as is): {document_title.split('_')[-1]}\n"
    prompt += f"<Form Fields>\n{keyval}\n</Form Fields>\n\n"
    with open("prompt.txt", "w") as f:
        f.write(prompt)
    
    def process_with_model(current_model):
        try:
            # Initialize messages list 
            messages = []
            
            # Add the current prompt
            messages.append({"role": "user", "content": prompt})
           
            print(f"Generating values using model: {current_model}")
            response = ollama.Client().chat(
                model=current_model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "seed": randint(0, 1000000),
                    "top_p": 1,
                    "top_k": 0,
                    "min_p": 0.05,
                },
                format="json",
            )
            generated_text = response["message"]["content"].strip()
            # Parse the JSON response
            results = json.loads(generated_text)

            # Update the text fields with the generated values
            filled_fields = 0
            for field in text_fields:
                value = results.get(field["name"], None)

                # Count filled fields
                if value is not None and value.strip() != "":
                    filled_fields += 1
                    
                    # Process Faker directives
                    if isinstance(value, str) and value.lower().startswith("faker:"):
                        faker_method = value.split(":", 1)[1].strip()
                        try:
                            # Get the method from faker and call it
                            faker_func = getattr(faker, faker_method)
                            field["value"] = faker_func()
                        except (AttributeError, TypeError) as e:
                            print(f"Error with Faker method '{faker_method}': {e}")
                            field["value"] = field["value"]  # Keep original value on error
                    else:
                        field["value"] = value
            
            # Calculate fill percentage
            fill_percentage = filled_fields / len(text_fields) if text_fields else 1.0
            
            return text_fields, fill_percentage
        except Exception as e:
            print(f"Error generating values with model {current_model}: {e}")
            return fields, 0.0
    
    # First attempt with the primary model
    updated_fields, fill_percentage = process_with_model(model_name)
    
    # If we have a fallback model and didn't get enough fields filled, try again
    if fallback_model and fill_percentage < min_fill_percentage:
        print(f"Only {fill_percentage:.1%} of fields were filled (minimum required: {min_fill_percentage:.1%})")
        print(f"Retrying with fallback model: {fallback_model}")
        updated_fields, new_fill_percentage = process_with_model(fallback_model)
        print(f"Fallback model filled {new_fill_percentage:.1%} of fields")
    
    return updated_fields


def main():
    parser = argparse.ArgumentParser(
        description="Generate form field values and update PDF forms"
    )
    parser.add_argument("input", help="Path to the input PDF template or directory containing PDFs")
    parser.add_argument(
        "input_csv",
        nargs="?",
        help="Path to the CSV with form field information (optional)",
    )
    parser.add_argument("-o", "--output", help="Path to the output PDF file or directory")
    parser.add_argument(
        "-m",
        "--model",
        default="llama3.2:latest",
        help="Ollama model to use (default: llama3.2:latest)",
    )
    parser.add_argument(
        "--fallback-model",
        help="Fallback model to use if primary model doesn't fill enough fields",
    )
    parser.add_argument(
        "--min-fill-percent",
        type=float,
        default=0.7,
        help="Minimum percentage of fields that must be filled (default: 0.7)",
    )
    parser.add_argument(
        "-n",
        "--variations",
        type=int,
        default=1,
        help="Number of variations to generate (default: 1)",
    )
    parser.add_argument(
        "-s",
        "--startvariation",
        type=int,
        default=1,
        help="The starting variation number (default: 1)",
    )

    parser.add_argument(
        "-d", "--directory-mode", 
        action="store_true",
        help="Process all PDF files in the input directory"
    )
    args = parser.parse_args()
    # Check if we're in directory mode
    # Generate new values page by page
    print(f"Generating new values with {args.model} model")
    if args.fallback_model:
        print(f"Using {args.fallback_model} as fallback if less than {args.min_fill_percent:.1%} of fields are filled")
    
    failed_logs = open("failed_logs.txt", "w")
    if args.directory_mode:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
            
        if not args.output or not os.path.isdir(args.output):
            print("Error: In directory mode, output must be a valid directory")
            return
            
        # Find all PDF files in the directory (non-recursive)
        pdf_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                    if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(args.input, f))]
        
        if not pdf_files:
            print(f"No PDF files found in {args.input}")
            return
            
        # Process each PDF file
        for pdf_file in pdf_files:
            print(f"\nProcessing {pdf_file}")
            try:
                process_pdf(pdf_file, args.input_csv, args.output,
                       args.model, args.variations, args.fallback_model, args.min_fill_percent, args.startvariation)
            except OCRException as e:
                print(f"Error processing {pdf_file}: {e}")
                failed_logs.write(f"{pdf_file}: {e}\n")
    else:
        # Process a single PDF file
        process_pdf(args.input, args.input_csv, args.output, 
                   args.model, args.variations, args.fallback_model, args.min_fill_percent)

    print("\nAll operations completed!")

def process_pdf(input_pdf, input_csv, output_dir, model_name, variations, fallback_model=None, min_fill_percent=0.6, start_variation=1):
    # Generate default output paths if not provided
    base_name = os.path.splitext(os.path.basename(input_pdf))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Handle case where input_csv is not provided
    if input_csv is None:
        # Try to find a CSV file with the same basename as the PDF
        pdf_dir = os.path.dirname(input_pdf)
        possible_csv_paths = [
            os.path.join(pdf_dir, f"{base_name}.csv"),
            os.path.join(pdf_dir, f"{base_name}_fields.csv"),
        ]

        for path in possible_csv_paths:
            if os.path.exists(path):
                input_csv = path
                print(f"Using CSV file: {input_csv}")
                break

        if input_csv is None:
            print(f"Error: No input CSV provided and no matching CSV file found for {input_pdf}.")
            print(f"Tried looking for: {', '.join(possible_csv_paths)}")
            return

    # Read form fields from CSV
    print(f"Reading form fields from {input_csv}")

    # Initialize variation hash
    current_hash = None
    fields_by_page = read_form_fields_from_csv(input_csv)

    if not fields_by_page:
        raise OCRException("No form fields found in the CSV")
    else:
        # Print all field names
        for page_key, fields in fields_by_page.items():
            print(f"Fields found on page {page_key}:")
            for field in fields:
                print(f"  {field['name']}")
    
    # Generate variations
    for variation in range(start_variation, variations + 1):
        # Generate or update hash for this variation
        should_redo = 10 # maximum number of retries
        while should_redo:
            try:
                current_hash = generate_hash(current_hash)

                # Determine output filenames for this variation
                if variations > 1:
                    variation_suffix = f"_var{variation}"
                else:
                    variation_suffix = ""

                # Set the output paths for this variation
                if output_dir:
                    if os.path.isdir(output_dir):
                        output_pdf_form = os.path.join(output_dir, f"{base_name}_form_{timestamp}{variation_suffix}.pdf")
                        output_pdf_flat = os.path.join(output_dir, f"{base_name}_flat_{timestamp}{variation_suffix}.pdf")
                        json_filename = os.path.join(output_dir, f"{base_name}_fields_{timestamp}{variation_suffix}.json")
                    else:
                        # For single file mode with specific output path
                        name, ext = os.path.splitext(output_dir)
                        output_pdf_form = f"{name}_form{variation_suffix}{ext}"
                        output_pdf_flat = f"{name}_flat{variation_suffix}{ext}"
                        json_filename = f"{name}_fields{variation_suffix}.json"
                else:
                    output_pdf_form = f"{base_name}_form_{timestamp}{variation_suffix}.pdf"
                    output_pdf_flat = f"{base_name}_flat_{timestamp}{variation_suffix}.pdf"
                    json_filename = f"{base_name}_fields_{timestamp}{variation_suffix}.json"

                print(
                    f"\nGenerating variation {variation} of {variations} with hash: {current_hash}"
                )

                updated_fields_by_page = {}
                for page_key, fields in fields_by_page.items():
                    print(f"Processing {page_key}")
                    # Generate form values with fallback model support
                    updated_fields = generate_form_values(
                        copy.deepcopy(fields), 
                        model_name, 
                        base_name,
                        fallback_model,
                        min_fill_percent
                    )
                    updated_fields_by_page[page_key] = updated_fields
                    
                # Create version 1: Update the PDF form fields (current behavior)
                update_pdf_form(input_pdf, updated_fields_by_page, output_pdf_form)
                print(f"Created editable PDF version at: {output_pdf_form}")
                # Create version 2: Write text directly onto the PDF
                fillpdf.fillpdfs.flatten_pdf(output_pdf_form, output_pdf_flat)
                print(f"Created flat PDF at: {output_pdf_flat}")
                ground_truth_fields = update_rects_to_actual_textbox(output_pdf_flat)
                images = fits_to_imgs(output_pdf_flat)
                data = to_dict(images, ground_truth_fields, output_pdf_flat)
                json.dump(data, open(json_filename, "w"), indent=2)
                print(f"Created JSON file at: {json_filename}")
                print(
                    f"Completed variation {variation} for {base_name}"
                )
                should_redo = 0
            except OCRException as e:
                print(f"Error with OCR: {e}")
                print(f"Redoing variation with new hash, attempts left: {should_redo}...")
                should_redo -= 1



if __name__ == "__main__":
    main()
