import csv

def read_form_fields_from_csv(csv_path) -> dict:
    """Read form field information from a CSV file."""
    # Record<page number, field>
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
