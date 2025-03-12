import re
import argparse
import os
import json
from pathlib import Path

from ultralytics import YOLO
from PIL import Image
import torch

from adapt import get_gt_arr, get_template_from_gt_arr, adapt_template_fields

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
                break

            img = images[page_no]

            adapted_fields = adapt_template_fields(template_fields, yolo, img)

            page_width = page['width']
            page_height = page['height']
            # Denormalize xywh values
            for field in adapted_fields:
                # Remove text from field
                del field['text']
                # Set text using ocr
                # field['text'] = ???
                field['x'] *= page_width
                field['y'] *= page_height
                field['w'] *= page_width
                field['h'] *= page_height

            adapted_template['fields'].extend(adapted_fields)

        adapted_templates.append(adapted_template)
        
    return adapted_templates


def overlay_template(template: dict, gt_arr: list[dict]) -> list[dict]:
    """
    Overlays template to gt instances
    - To be used for non-adapted template evaluation
    - XYWH VALUES FROM `template` MUST BE NORMALIZED!!!
    - The output is denormalized based on the gt per_page data
    - Returns an overlayed template per gt instance
    """

    overlayed_templates = []

    for gt in gt_arr:

        images = gt['images']
        per_page = gt['per_page']

        overlayed_template = gt.copy()
        overlayed_template['fields'] = []

        for page_no, page in enumerate(per_page):

            template_fields = [
                field for field in template['fields'] if field['page'] == page_no
            ]

            if not template_fields:
                break

            overlayed_fields = [field.copy() for field in template_fields]

            page_width = page['width']
            page_height = page['height']
            for field in overlayed_fields:
                # Remove text from field
                del field['text']
                # Set text using ocr
                # field['text'] = ???
                # Denormalize xywh values
                field['x'] *= page_width
                field['y'] *= page_height
                field['w'] *= page_width
                field['h'] *= page_height

            overlayed_template['fields'].extend(overlayed_fields)

        overlayed_templates.append(overlayed_template)

    return overlayed_templates

def to_coco(gt_arr: dict, output_dir: str = None) -> dict:
    """
    Converts gt list format to COCO format
    - optionally saves the COCO dataset if output_dir is provided

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

    if output_dir is not None:
        # If output_dir is specified, create the folder for the dataset
        output_path = Path(output_dir)
        images_path = output_path / "images"
        (images_path).mkdir(parents=True, exist_ok=True)

    images = []
    categories = []
    annotations = []

    # Dict to store category_name -> category_id
    category_ids = {}

    for gt in gt_arr:

        pdf_path = Path(gt['filename'])
        pdf_images = gt['images']

        fields = gt['fields']
        per_page = gt['per_page']

        # Dict to store page_no -> image_id
        image_ids = {}

        for page_no, page in enumerate(per_page):
            page_width = page['width']
            page_height = page['height']

            image_id = len(images)
            image_file_name = f"{pdf_path.stem}_{page_no}.jpg"

            images.append({
                'id': image_id,
                'width': page_width,
                'height': page_height,
                'file_name': image_file_name
            })

            if output_dir is not None:
                # If output dir is specified, save the image
                image_path = images_path / image_file_name
                image = pdf_images[page_no]
                image.save(image_path)
            
            image_ids[page_no] = image_id
            
        for field in fields:
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

            page_no = field['page']
            image_id = image_ids[page_no]

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
                "score": 1 # confidence score = 1, since adaptation handles the threshold
            })

    coco_dataset = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    if output_dir is not None:
        # If output_dir is specified, save the dataset to the folder
        with open(output_path / "coco.json", "w") as f:
            json.dump(coco_dataset, f)

    return coco_dataset


def generate_datasets(template: dict, gt_arr: list[dict], yolo: YOLO, output_dir: str = None) -> tuple[dict]:
    """
    Generates coco gt, adapted dt, and overlayed dt datasets from template and gt_arr 
    - optionally saves the COCO datasets if output_dir is provided
    - XYWH VALUES FROM `template` MUST BE NORMALIZED!!!
    """

    # Adapted templates
    adapted_templates = adapt_template(template, gt_arr, yolo)

    # Non-adapted templates
    overlayed_templates = overlay_template(template, gt_arr)

    gt_output_dir = None
    adapted_output_dir = None
    overlayed_output_dir = None
    if output_dir is not None:
        gt_output_dir = str(Path(output_dir) / 'gt')
        adapted_output_dir = str(Path(output_dir) / 'adapted_dt')
        overlayed_output_dir = str(Path(output_dir) / 'overlayed_dt')

    coco_gt = to_coco(gt_arr, gt_output_dir)
    adapted_dt = to_coco(adapted_templates, adapted_output_dir)
    overlayed_dt = to_coco(overlayed_templates, overlayed_output_dir)

    return coco_gt, adapted_dt, overlayed_dt



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

    coco_gt, adapted_dt, overlayed_dt = generate_datasets(template, gt_arr, yolo, args.output_dir)

    print(coco_gt)
    print(adapted_dt)
    print(overlayed_dt)

if __name__ == "__main__":
    main()

