import re
import argparse
import os
import json

from ultralytics import YOLO
from PIL import Image
import torch
import torchvision.ops as ops

from ocr import fits_to_imgs

def get_gt_arr(gt_dir: str) -> list[dict]:
    '''
    Get all gt as a list with images attached
    '''

    gt_arr = []

    # Find all PDF files in the directory (non-recursive)
    pdf_files = [os.path.join(gt_dir, f) for f in os.listdir(gt_dir) 
                if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(gt_dir, f))]

    # For each PDF file, get its corresponding gt data
    for pdf_file in pdf_files:
        json_file = re.sub('.pdf$', '.json', pdf_file)

        with open(json_file, 'r') as f:
            gt_data = json.load(f)

        gt_data['images'] = fits_to_imgs(pdf_file)

        gt_arr.append(gt_data)
                
    return gt_arr

def get_template_from_gt_arr(gt_arr: list[dict]) -> dict:
    '''
    Create a template based on the given gt list
    '''

    template_fields = []
    found_fields = {}

    for gt in gt_arr:

        per_page = gt['per_page']

        for field in gt['fields']:
            field_name = field['name']

            if (field_name not in found_fields):
                found_fields[field_name] = True

                field_page = field['page']

                page_data = per_page[field_page]
                page_width = page_data['width']
                page_height = page_data['height']
                
                field_copy = field.copy()
                # Remove text
                del field_copy['text']
                # Normalize xywh values
                field_copy['x'] /= page_width
                field_copy['y'] /= page_height
                field_copy['w'] /= page_width
                field_copy['h'] /= page_height

                template_fields.append(field_copy)
    
    template = {
        'fields': template_fields
    }

    return template

def adapt_template_fields(template_fields: list[dict], yolo: YOLO, img: Image.Image) -> dict:
    '''
    Adapt the template fields to an image (document).

    NOTE: Template fields can be mapped to multiple document fields
          But each document field may only be mapped to 0 or 1 template field
    Template field -> 0 to many Document fields
    Document field -> 0 to 1 Template field
    '''

    # step 1: Use YOLO to get document fields (boxes)
    document_boxes = yolo.predict(source=[img], imgsz=1024, conf=0.25)[0].boxes
    
    # step 2: Adapt template fields with document fields

    # Get xyxy tensors of document boxes and template boxes
    d_xyxyn = document_boxes.xyxyn
    t_xyxyn = torch.tensor(
        [
            [
                t_field['x'],
                t_field['y'],
                t_field['x'] + t_field['w'],
                t_field['y'] + t_field['h']
            ]
            for t_field in template_fields
        ],
        dtype=torch.float32,
        device=document_boxes.xyxyn.device
    )

    # Get adapted boxes from document boxes and template boxes
    a_xyxyn, adapted_count = adapt_boxes(d_xyxyn, t_xyxyn)

    # Convert adapted boxes to fields
    adapted_fields = [t_field.copy() for t_field in template_fields]
    for a_id, a_field in enumerate(adapted_fields):
        # Update bbox values
        a_box = a_xyxyn[a_id]
        x = a_box[0].item()
        y = a_box[1].item()
        w = a_box[2].item() - x
        h = a_box[3].item() - y
        a_field['x'] = x
        a_field['y'] = y
        a_field['w'] = w
        a_field['h'] = h

    # Print analytics
    adapted_fields_count = torch.count_nonzero(adapted_count).item()
    print("---- Start of Analytics ----")
    print(f"Template fields (T): {len(template_fields)}")
    print(f"Detected fields (D): {len(document_boxes)}")
    print(f"Adapted fields (A): {adapted_fields_count}")
    print(f"Ratio A/T: {round(adapted_fields_count/len(template_fields),2)}")
    print(f"Ratio A/D: {round(adapted_fields_count/len(document_boxes),2)}")
    print("---- End of Analytics ----")
    
    return adapted_fields

def adapt_boxes(d_xyxyn:torch.Tensor, t_xyxyn:torch.Tensor):
    """
    Get adapted boxes from document and template boxes
    - d_xyxy and t_xyxy must be tensors containing boxes in in xyxy format
    - Returns two values:
        - a_xyxy: a tensor of adapted boxes in xyxy format
        - adapted_count: a tensor containing the number of times each template box in adapted
    """
    # The xyxy tensor for the adapted boxes
    a_xyxyn = t_xyxyn.clone()
    # A tensor that records the no. of times a template field is adapted
    adapted_count = torch.zeros(t_xyxyn.size(0), dtype=torch.int, device=t_xyxyn.device)

    # Get IoU matrix
    iou_matrix = ops.box_iou(d_xyxyn, t_xyxyn)
    # Get the template boxes with the highest iou for each document box
    closest_t_ious, closest_t_ids = iou_matrix.max(dim=1)

    # For each document box pair, expand the adapted box to include the document box
    MIN_IOU_THRESHOLD = 0.5
    for d_id, (t_id, iou) in enumerate(zip(closest_t_ids, closest_t_ious)):
        # Adapt only when IoU > threshold
        if iou > MIN_IOU_THRESHOLD:
            d_box = d_xyxyn[d_id]

            if adapted_count[t_id] <= 0: # On first match, overwrite the bbox
                a_xyxyn[t_id] = d_box
            else: # Next matches, expand the bbox
                a_box = a_xyxyn[t_id]
                a_xyxyn[t_id, 0] = torch.min(d_box[0], a_box[0])  # x1
                a_xyxyn[t_id, 1] = torch.min(d_box[1], a_box[1])  # y1
                a_xyxyn[t_id, 2] = torch.max(d_box[2], a_box[2])  # x2
                a_xyxyn[t_id, 3] = torch.max(d_box[3], a_box[3])  # y2

            adapted_count[t_id] += 1

    return a_xyxyn, adapted_count

def main():
    parser = argparse.ArgumentParser(
        description="Adapt a template (constructed from gt) to each gt"
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

    args = parser.parse_args()

    yolo = YOLO(args.model, task="detect")

    gt_arr = get_gt_arr(args.input_dir)

    template = get_template_from_gt_arr(gt_arr)

    adapted_templates = []

    for gt in gt_arr:

        images = gt['images']
        per_page = gt['per_page']

        for page, page_data in enumerate(per_page):

            template_fields = [
                field for field in template['fields'] if field['page'] == page
            ]

            if not template_fields:
                break

            img = images[page]

            adapted_fields = adapt_template_fields(template_fields, yolo, img)

            page_width = page_data['width']
            page_height = page_data['height']
            # Set text using ocr
            # field['text'] = ???
            # Denormalize xywh values
            for field in adapted_fields:
                field['x'] *= page_width
                field['y'] *= page_height
                field['w'] *= page_width
                field['h'] *= page_height

        adapted_templates.append({
            'filename': gt['filename'],
            'fields': adapted_fields,
            'per_page': per_page
        })

    print(adapted_templates)

    return

if __name__ == "__main__":
    main()