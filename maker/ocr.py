import json
import os
from PIL import Image
import fitz
from pytesseract import image_to_data
from PIL import ImageDraw
import argparse

TARGET_DPI = 300
# According to the documentation, this is the default DPI used by fitz
DEFAULT_FITZ_DPI = 72


def fits_to_imgs(pdf_path: str) -> list[Image.Image]:
    doc = fitz.open(pdf_path)
    return [page_to_img(doc[i]) for i in range(len(doc))]


def page_to_img(page: fitz.Page) -> Image.Image:
    pix = page.get_pixmap(dpi=TARGET_DPI)  # type: ignore
    img = pix.pil_image()
    return img


def get_scaled_rect(widget, dpi):
    """
    Calculates the scaled rectangle of a widget in a PDF given a DPI.

    Args:
        widget (fitz.Widget): The widget object.
        dpi (int or float): The desired DPI.

    Returns:
        fitz.Rect: The scaled rectangle.
    """

    rect = widget.rect  # Original rectangle in PDF units

    # Get the transformation matrix from PDF units to pixels at the given DPI.
    matrix = fitz.Matrix(dpi / DEFAULT_FITZ_DPI, dpi / DEFAULT_FITZ_DPI)

    # Apply the matrix to the rectangle to get the scaled rectangle.
    scaled_rect = rect * matrix

    return scaled_rect


class OCRException(Exception):
    pass
def update_rects_to_actual_textbox(pdf_path):
    """Note: this will throw if a `data` is empty"""
    doc = fitz.open(pdf_path)
    data_per_page = {}
    result = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        data_per_page[page_num] = []
        orig_cropbox = page.cropbox
        annotations = page.widgets()
        for annot in annotations:
            if annot.field_type_string == "Text":
                # Get the original annotation rectangle coordinates
                annot_rect = annot.rect
                page.set_cropbox(annot_rect)
                pix: fitz.Pixmap = page.get_pixmap(dpi=TARGET_DPI)  # type: ignore
                img: Image.Image = pix.pil_image()
                data = image_to_xywht(img)
                page.set_cropbox(orig_cropbox)
                # Get the relative bbox within the cropped image
                if data:
                    scaled_rect = get_scaled_rect(annot, TARGET_DPI)
                    text = implode_boxes(data)
                    if (text == ''):
                        print("Empty value found in", annot.field_name)
                        continue
                    try:
                        real_text_bbox = enclose_boxes(data)
                    except Exception as e:
                        raise OCRException(f"Error processing {annot.field_name} {data}") from e
                    # Adjust coordinates to be relative to the entire page
                    adjusted_bbox = (
                        real_text_bbox[0] + scaled_rect.x0,  # type: ignore
                        real_text_bbox[1] + scaled_rect.y0,  # type: ignore
                        real_text_bbox[2],
                        real_text_bbox[3],
                    )

                    result.append(
                        {
                            "page": page_num,
                            "name": annot.field_name,
                            # We're using the OCR'd text because the LLM may generate text that would get cut off when flattening. If we use the original text, our score will be lower.
                            "text": text,
                            "x": adjusted_bbox[0],
                            "y": adjusted_bbox[1],
                            "w": adjusted_bbox[2],
                            "h": adjusted_bbox[3],
                        }
                    )
                else:
                    print(f"No text found in {annot.field_name} on page {page_num}")

    return result

def implode_boxes(xywht: list[tuple]) -> str:
    """Implode a list of boxes into a single string."""
    return " ".join([box[4] for box in xywht if box[4] != '']).strip()
 
def enclose_boxes(xywht: list[tuple]) -> tuple:
    """Enclose a list of boxes within a bounding box.""" 
    # return (min_x, min_y, width, height)
    # find max and min x and y that would enclose all the boxes
    # filter empty text boxes
     # will throw if xywht is empty
    xywht = [box for box in xywht if box[4] != '']
    min_x = min([box[0] for box in xywht])
    min_y = min([box[1] for box in xywht])
    max_x = max([box[0] + box[2] for box in xywht])
    max_y = max([box[1] + box[3] for box in xywht])
    return (
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
    )


def image_to_xywht(image: Image.Image) -> list[tuple]:
    data = image_to_data(image, output_type="dict")
    n_boxes = len(data["level"])
    xywht = []
    for i in range(n_boxes):
        xywht.append(
            (
                data["left"][i],
                data["top"][i],
                data["width"][i],
                data["height"][i],
                data["text"][i],
            )
        )
    return xywht


# Preview the image w/ the bboxes
def plot_boxes(image: Image.Image, fields: list[dict], page_num: int):
    """Preview the image with the bounding boxes"""

    for field in fields:
        if field["page"] != page_num:
            continue
        x, y, w, h = field["x"], field["y"], field["w"], field["h"]
        draw = ImageDraw.Draw(image)
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
    return image


def to_dict(images: list[Image.Image], data: list[dict], pdf_path: str):
    """Convert the data to a JSON format"""
    return {
        # basename
        "filename": os.path.basename(pdf_path),
        "fields": data,
        # get width and height per page
        "per_page": [{"width": img.width, "height": img.height} for img in images],
    }

def process_pdf(pdf_path, output_dir=None, show_image=False):
    """Process a single PDF file and generate JSON output"""
    print(f"Processing {pdf_path}")
    try:
        adjusted = update_rects_to_actual_textbox(pdf_path)
        images = fits_to_imgs(pdf_path)
        
        # Determine output JSON path
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, f"{base_name}.json")
        else:
            json_path = f"{base_name}.json"
        
        # Save JSON data
        json.dump(to_dict(images, adjusted, pdf_path), open(json_path, "w"), indent=2)
        print(f"Created JSON file at: {json_path}")
        
        # Show image with boxes if requested
        if show_image and images:
            img = plot_boxes(images[0], adjusted, 0)
            img.show()
            
    except OCRException as e:
        print(f"Error processing {pdf_path}: {e}")
    except Exception as e:
        print(f"Unexpected error processing {pdf_path}: {e}")

def main():
    """Run if main module"""
    parser = argparse.ArgumentParser(
        description="Process PDFs and generate JSON files with OCR data"
    )
    parser.add_argument("input", help="Path to the input PDF file or directory containing PDFs")
    parser.add_argument("-o", "--output", help="Path to the output directory for JSON files")
    parser.add_argument("-s", "--show", action="store_true", help="Show images with bounding boxes")
    args = parser.parse_args()
    
    # Check if input is a directory or a file
    if os.path.isdir(args.input):
        # Process all PDFs in directory
        pdf_files = [os.path.join(args.input, f) for f in os.listdir(args.input) 
                     if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(args.input, f))]
        
        if not pdf_files:
            print(f"No PDF files found in {args.input}")
            return
            
        for pdf_path in pdf_files:
            process_pdf(pdf_path, args.output, args.show)
    else:
        # Process a single PDF file
        process_pdf(args.input, args.output, args.show)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
