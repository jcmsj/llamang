import re
import argparse
import os
import json
from pathlib import Path

from ultralytics import YOLO
from PIL import Image
import torch

from adapt import get_gt_arr, get_template_from_gt_arr, adapt_template_fields
from ocr import image_to_xywht, implode_boxes

def adapt_template(template: dict, gt_arr: list[dict], yolo: YOLO) -> list[dict]:
    """
    Adapts template to gt instances
    - To be used for adapted template evaluation
    - XYWH VALUES FROM `template` MUST BE NORMALIZED!!!
    - The output is denormalized based on the gt per_page data
    - Returns an adapted template per gt instance
    """

    adapted_templates = []

    for gt in gt_arr:

        images = gt['images']
        per_page = gt['per_page']

        adapted_template = gt.copy()
        adapted_template['fields'] = []

        for page_no, page in enumerate(per_page):

            template_fields = [
                field for field in template['fields'] if field['page'] == page_no
            ]

            if not template_fields:
                continue

            img = images[page_no]

            adapted_fields = adapt_template_fields(template_fields, yolo, img)

            page_width = page['width']
            page_height = page['height']
            for field in adapted_fields:
                # Denormalize xywh values
                field['x'] *= page_width
                field['y'] *= page_height
                field['w'] *= page_width
                field['h'] *= page_height

                # Set text using ocr
                x1, y1, w, h = field['x'], field['y'], field['w'], field['h']
                x2, y2 = x1 + w, y1 + h
                field_img = img.crop((x1, y1, x2, y2))
                field['text'] = implode_boxes(image_to_xywht(field_img))

            adapted_template['fields'].extend(adapted_fields)

        adapted_templates.append(adapted_template)
        
    return adapted_templates


def overlay_template(template: dict, gt_arr: list[dict]) -> list[dict]:
    """
    Overlays template to gt instances
    - To be used for non-adapted template evaluation
    - XYWH VALUES FROM `template` MUST BE NORMALIZED!!!
    - The output is denormalized based on the gt per_page data
    - Returns an overlaid template per gt instance
    """

    overlaid_templates = []

    for gt in gt_arr:

        images = gt['images']
        per_page = gt['per_page']

        overlaid_template = gt.copy()
        overlaid_template['fields'] = []

        for page_no, page in enumerate(per_page):

            template_fields = [
                field for field in template['fields'] if field['page'] == page_no
            ]

            if not template_fields:
                continue

            img = images[page_no]

            overlaid_fields = [field.copy() for field in template_fields]

            page_width = page['width']
            page_height = page['height']
            for field in overlaid_fields:
                # Denormalize xywh values
                field['x'] *= page_width
                field['y'] *= page_height
                field['w'] *= page_width
                field['h'] *= page_height

                # Set text using ocr
                x1, y1, w, h = field['x'], field['y'], field['w'], field['h']
                x2, y2 = x1 + w, y1 + h
                field_img = img.crop((x1, y1, x2, y2))
                field['text'] = implode_boxes(image_to_xywht(field_img))

            overlaid_template['fields'].extend(overlaid_fields)

        overlaid_templates.append(overlaid_template)

    return overlaid_templates

def to_coco(gt_arr: dict, images = [], image_ids = {}, categories = [], category_ids = {}, image_data = []) -> dict:
    """
    Converts gt list format to COCO format
    - NOTE: When working with shared objects, this does not make copies!

    gt format:
    [
        {
            filename: str,
            fields: [
                {
                    name: str,
                    text: str,
                    x: number,
                    y: number,
                    w: number,
                    h: number
                }, ...
            ],
            per_page: [
                {
                    width: number,
                    height: number
                }, ...
            ],
            images: [  
                img: PIL.Image.Image, ...
            ]
        }, ...
    ]

    COCO format:
    {
        "images": [
            {
                "id": number,
                "width": number,
                "height": number,
                "file_name": str
            }, ...
        ],
        "annotations": [
            {
                "id": number,
                "image_id": number,
                "category_id": number,
                "bbox": [
                    x1: number,
                    y1: number,
                    x2: number,
                    y2: number
                ],
                "area": number,
                "iscrowd": number,
            }, ...
        ],
        "categories": [
            {
                "id": number,
                "name": str
            }, ...
        ],
    }
    """

    annotations = []

    for gt in gt_arr:

        pdf_path = Path(gt['filename'])
        pdf_images = gt['images']

        fields = gt['fields']
        per_page = gt['per_page']

        for page_no, page in enumerate(per_page):

            page_fields = [field for field in fields if field['page'] == page_no and field['text']]

            if not page_fields:
                continue

            page_width = page['width']
            page_height = page['height']

            image_file_name = f"{pdf_path.stem}_{page_no}.jpg"

            if image_file_name in image_ids:
                image_id = image_ids[image_file_name]
            else:
                image_id = len(images)
                image_ids[image_file_name] = image_id

                images.append({
                    'id': image_id,
                    'width': page_width,
                    'height': page_height,
                    'file_name': image_file_name
                })
                image_data.append(pdf_images[page_no])
            
            for field in page_fields:
                category_name = field['name']

                if category_name in category_ids:
                    category_id = category_ids[category_name]
                else:
                    category_id = len(categories)
                    category_ids[category_name] = category_id

                    categories.append({
                        'id': category_id,
                        'name': category_name
                    })

                annotation_id = len(annotations)

                x1, y1 = field['x'], field['y']
                w, h = field['w'], field['h']
                x2, y2 = w + x1, h + y1
                
                annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x1, y1, x2, y2],
                    "area": w*h,
                    "iscrowd": 0,
                    "score": 1, # confidence score = 1, since adaptation handles the threshold
                    "text": field['text'] # custom attribute for the ocr text
                })

    coco_dataset = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    return coco_dataset


def generate_datasets(template: dict, gt_arr: list[dict], yolo: YOLO, output_dir: str = None) -> tuple[dict]:
    """
    Generates coco gt, adapted dt, and overlaid dt datasets from template and gt_arr
    - optionally saves the COCO datasets if output_dir is provided
    - XYWH VALUES FROM `template` MUST BE NORMALIZED!!!
    """

    print('Adapting templates...')
    # Adapted templates
    adapted_templates = adapt_template(template, gt_arr, yolo)

    print('Overlaying templates...')
    # Non-adapted templates
    overlaid_templates = overlay_template(template, gt_arr)

    # Shared objects between datasets

    # Image list
    images = []
    # Dict to store image_file_name -> image_id
    image_ids = {}
    # Category list
    categories = []
    # Dict to store category_name -> category_id mappings
    category_ids = {}
    # Image data list
    image_data = []

    print('Converting to COCO datasets...')
    coco_gt = to_coco(gt_arr, images, image_ids, categories, category_ids, image_data)
    adapted_dt = to_coco(adapted_templates, images, image_ids, categories, category_ids, image_data)
    overlaid_dt = to_coco(overlaid_templates, images, image_ids, categories, category_ids, image_data)

    # If output_dir is specified, save the images and the dataset json files
    if output_dir is not None:
        output_path = Path(output_dir)
        images_path = output_path / "images"

        # Create output directory
        print('Creating output directory...')
        (images_path).mkdir(parents=True, exist_ok=True)

        # Save COCO json files
        print('Saving COCO json files...')
        with open(Path(output_dir) / 'coco_gt.json', "w") as f:
            json.dump(coco_gt, f)
        with open(Path(output_dir) / 'adapted_dt.json', "w") as f:
            json.dump(adapted_dt, f)
        with open(Path(output_dir) / 'overlaid_dt.json', "w") as f:
            json.dump(overlaid_dt, f)

        # Save images
        print('Saving images...')
        for image in images:
            image_path = images_path / image['file_name']
            img = image_data[image['id']]
            img.save(image_path)

    return coco_gt, adapted_dt, overlaid_dt



def main():
    parser = argparse.ArgumentParser(
        description="Convert gt files to coco datasets (ground truth, adapted templates, and non-adapted templates)"
    )

    parser.add_argument(
        "-i", "--input-dir",
        required=True,
        help="Path of the input directory where the gt is stored"
    )

    parser.add_argument(
        "-m", "--model",
        required=True,
        help="Path to the model to be used for detection (.onnx / .pt)"
    )

    parser.add_argument(
        "-o", "--output-dir",
        required=True,
        help="Output directory where to save the coco-formatted dataset including images"
    )

    args = parser.parse_args()

    yolo = YOLO(args.model, task="detect")

    gt_arr = get_gt_arr(args.input_dir)

    # NOTE: xywh values from get_template_from_gt_arr is normalized!
    template = get_template_from_gt_arr(gt_arr)

    coco_gt, adapted_dt, overlaid_dt = generate_datasets(template, gt_arr, yolo, args.output_dir)

    print('COCO datasets saved in: ' + args.output_dir)

if __name__ == "__main__":
    main()

