import fitz


def update_pdf_form(pdf_path:str, fields_by_page:dict[str,dict], output_path:str):
    """Update the PDF form with the generated values."""
    doc = fitz.open(pdf_path)

    for page_key, fields in fields_by_page.items():
        # Extract page number from the key (e.g., "page_1" -> 0)
        try:
            page_num = int(page_key.split('_')[1]) - 1
        except Exception:
            page_num = 0

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
