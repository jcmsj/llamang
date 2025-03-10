import csv


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
