import argparse
import json
from pdfrw import PdfReader, PdfWriter, PdfName, PdfString, PdfDict, PdfObject
import ollama

def fill_pdf_form(input_pdf, output_pdf, field_values):
    """Fills form fields in a PDF with provided values."""
    pdf = PdfReader(input_pdf)
    for page in pdf.pages:
        if page["/Annots"]:
            for annot in page["/Annots"]:
                if annot["/Subtype"] == "/Widget":
                    # Handle fields that use Parent/Kids structure
                    if not annot["/T"]:
                        annot = annot["/Parent"]
                    if annot["/T"]:
                        key = annot["/T"].to_unicode()
                        if key in field_values:
                            pdfstr = PdfString.encode(str(field_values[key]))
                            annot.update(PdfDict(V=pdfstr))

    # Ensure fields display properly
    if pdf.Root.AcroForm:
        pdf.Root.AcroForm.update(PdfDict(NeedAppearances=PdfObject("true")))

    writer = PdfWriter()
    writer.write(output_pdf, pdf)

def load_example_values(json_path):
    """Load example values from a JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return {}

def analyze_and_generate_values(input_pdf, example_values, ollama_model):
    """Analyze PDF fields, use examples, and generate missing values."""
    pdf = PdfReader(input_pdf)
    all_fields = []
    
    # Extract all field names from PDF
    for page in pdf.pages:
        if page["/Annots"]:
            for annot in page["/Annots"]:
                if annot["/Subtype"] == "/Widget":
                    # Handle fields that use Parent/Kids structure
                    if not annot["/T"]:
                        annot = annot["/Parent"]
                    if annot["/T"]:
                        field_name = annot["/T"].to_unicode()
                        if field_name not in all_fields:
                            all_fields.append(field_name)
                        else:
                            # count the number of times the field name appears, suffix a (n) 
                            # where n is the number of times it appears
                            all_fields.append(f"{field_name} ({all_fields.count(field_name)})")
    
    # Separate example fields and empty fields
    example_fields = {}
    empty_fields = []
    
    for field in all_fields:
        if field in example_values and example_values[field] and example_values[field] != "0" and example_values[field] != "0.00":
            example_fields[field] = example_values[field]
        else:
            empty_fields.append(field)
    
    # Generate values for empty fields using examples as context
    field_types = {}
    for field in empty_fields:
        field_types[field] = determine_data_type_ollama(field, ollama_model)
        print(f"[OLLAMA] Field: {field} - Type: {field_types[field]}")
    
    # Generate values using examples as context
    generated_values = generate_values_with_examples(
        empty_fields, 
        field_types, 
        example_fields,
        ollama_model
    )
    
    # Combine example values with generated values
    final_values = {**example_values, **generated_values}
    
    # Print the final values
    print("\nFinal field values:")
    for field, value in final_values.items():
        if field in generated_values:
            print(f"{field} -> {value} (Generated)")
        elif field in example_fields:
            print(f"{field} -> {value} (Example)")
        else:
            print(f"{field} -> {value}")
    
    return final_values

def determine_data_type_ollama(field_name, ollama_model):
    """Determine the type of data to generate using Ollama."""
    try:
        response = ollama.chat(
            model=ollama_model,
            messages=[
                {
                    "role": "user",
                    "content": f"""Classify the following form field name into one of these categories: identifier, name, address, currency, phrase, integer, email, phone, date, or string. Field name: {field_name}. Respond with only the category name. If the field name doesn't match any of these categories, respond with 'string'.""",
                },
            ],
        )
        return response["message"]["content"].strip().lower()
    except Exception as e:
        print(f"Error during Ollama interaction: {e}. Defaulting to string.")
        return "string"

def generate_values_with_examples(empty_fields, field_types, example_fields, ollama_model):
    """Generate values for empty fields using example fields as context."""
    try:
        # Build a structured description of the form
        form_description = "Form fields to generate:\n"
        for field in empty_fields:
            form_description += f"- {field} (Type: {field_types[field]})\n"
        
        examples_description = "Example values from the form:\n"
        for field, value in example_fields.items():
            examples_description += f"- {field}: {value}\n"
        
        # Create the prompt
        prompt = f"""You are tasked with generating realistic sample data for a form with empty fields.

        <Form Fields That Need Values>
        {form_description}
        </Form Fields>
        
        <Example Values From Other Fields>
        {examples_description}
        </Example Values>

        Instructions:
        1. Generate appropriate data for each empty field, ensuring consistency with the example values.
        2. Maintain coherence with the existing examples (like using the same person's details or business context).
        3. For currency fields, use the same format as any example currency values.
        4. For date fields, follow the date format shown in examples.
        5. All generated values should appear to be from the same context/document.
        
        Format your response as a JSON object with field names as keys and your generated values as strings.
        Return only the JSON object without any additional text."""

        # Get the response from Ollama
        response = ollama.chat(
            model=ollama_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        
        try:
            return json.loads(response["message"]["content"])
        except json.JSONDecodeError:
            # Try to extract just the JSON part if the model added explanations
            content = response["message"]["content"]
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            raise
            
    except Exception as e:
        print(f"Error generating data with examples: {e}. Falling back to individual generation.")
        return fallback_individual_generation(empty_fields, field_types, example_fields, ollama_model)

def fallback_individual_generation(empty_fields, field_types, example_fields, ollama_model):
    """Generate values for each empty field individually, with examples as context."""
    generated_values = {}
    
    for field in empty_fields:
        data_type = field_types[field]
        
        prompt = f"""Generate a realistic {data_type} value for a form field named '{field}'.
        
        The form already has these filled values:
        {json.dumps(example_fields, indent=2)}
        
        Make sure your generated value is consistent with the existing values and context.
        Return only the value as a string, with no explanations."""
        
        try:
            response = ollama.chat(
                model=ollama_model,
                messages=[{"role": "user", "content": prompt}],
            )
            generated_values[field] = response["message"]["content"].strip().strip('"').strip("'")
        except Exception as e:
            print(f"Error generating {field}: {e}. Using default.")
            generated_values[field] = f"Sample {data_type}"
    
    return generated_values

def main():
    parser = argparse.ArgumentParser(
        description="Fill PDF form fields using examples and LLM-generated data."
    )
    parser.add_argument("input_pdf", help="Path to the input PDF file.")
    parser.add_argument("output_pdf", help="Path to the output PDF file.")
    parser.add_argument("json_file", help="Path to JSON file with example values.")
    parser.add_argument(
        "--ollama-model",
        default="llama3.2:latest",
        help="Ollama model to use (default: llama3.2:latest).",
    )

    args = parser.parse_args()
    
    # Load example values from JSON
    example_values = load_example_values(args.json_file)
    print(f"Loaded {len(example_values)} fields from {args.json_file}")
    
    # Analyze and generate values
    final_values = analyze_and_generate_values(
        args.input_pdf, 
        example_values, 
        args.ollama_model
    )
    
    # Fill the PDF
    fill_pdf_form(args.input_pdf, args.output_pdf, final_values)
    print(f"Filled PDF saved to {args.output_pdf}")

if __name__ == "__main__":
    main()
