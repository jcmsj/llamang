import asyncio
import re
import pandas as pd
import pytesseract
from ollama import AsyncClient
from cv2.typing import MatLike
import preprocess
from preview import preview

from phrase_bbox import include_bboxes
# from search import include_bboxes

def perform_ocr(image: MatLike) -> pd.DataFrame:
    """Perform OCR and return raw Tesseract data"""
    print("Opening image...")

    print("Performing OCR...")
    detailed_data = pytesseract.image_to_data(
        image, output_type=pytesseract.Output.DICT
    )
    results = []
    for i in range(len(detailed_data["text"])):
        if detailed_data["text"][i].strip():
            results.append({
                "text": detailed_data["text"][i],
                "x1": detailed_data["left"][i],
                "y1": detailed_data["top"][i],
                "x2": detailed_data["left"][i] + detailed_data["width"][i],
                "y2": detailed_data["top"][i] + detailed_data["height"][i],
            })

    return pd.DataFrame(results)


async def proces_with_llm(df: pd.DataFrame) -> list[dict]:
    """Process OCR results with deepseek model"""
    client = AsyncClient()
    accumulated_response = ""

    # build the full text input, add spaces, but also try to restore the newlines, by comparing the previous text's bbox w/ the current text's bbox, if they do not overlap, add a newline
    text =""
    for i, row in df.iterrows():
        if i == 0:
            text += row["text"]
        else:
            prev_row = df.iloc[i-1] # type: ignore
            line_height = row["y2"] - row["y1"]
            if prev_row["y2"] < row["y1"]:
                text += "\n" + row["text"]
            elif row["x1"] - prev_row["x2"] > line_height*1.5:
                text += "\t" + row["text"]
            else:
                text += " " + row["text"]



    print("TEXT", text)

    async for chunk in await client.chat(
        model="llama3.2:latest",
        # model="deepseek-r1:1.5b", # prompt is still incorrect with this model
        options={
            "temperature": 0.2, # configures the model's creativity
        },
        messages=[
            {
                "role": "system",
                "content": f"""
                   TASK: Extract relevant information from a document using the ocr results in the form of key:value pairs.

                STEPS:
                1. **Identify form field label**: look for short phrases or words that are likely to represent a field name in a form or document. These are usually the labels or titles of the fields. Prefer shorter phrases less than 5 words.

                2. **Identify form field values**: look for short phrases or words that are likely to represent the values of the fields such as named entities. This includes things like names, organizations, locations, dates, times, products, units, currencies, subjects, events, and more. These are important to the context of the document. These are usually the actual information associated with the field label.

                3. **pairing** Make key-value pairs of the form field label and the corresponding field value. Identify at least 20 of these pairs.

                4. **Object schema**: Each entry should be a dictionary with the following keys:
                    ``` {{"key": "field label", "value": "the field value"}}```

                5 **Ensure unique keys**: Ensure unique keys for each entity type, avoiding reuse of the same key for different values.

                6 **Table structure**: Try to identify tables and their structure using common delimeters and consistent spacing. Their headers or columns could be the field labels.
                7. **Final output**: 
                    - Format the output as CSV with two columns: key,value
                        - where key is the identified entity and value is the extracted value
                    - First line must be the header: key,value
                    - Properly escape commas and quotes in values
                    - the keys should be human-readable and not too long. E.g. if you found an entity "Full name", use that key as it is, don't apply naming conventions.
                    - Example format:
                        ```csv
                        key,value
                        name,John Smith
                        amount,42
                        address,"123 Main St, Suite 100"
                        height: "5'11\"" the double quotes are escaped
                        ```
                    - Wrap the output in CSV code block for easy parsing: ```csv ... ```
                """,
            },
            {"role": "user", "content": f"<ocr-input>{text}</ocr-input>"},
        ],
        stream=True,
    ):
        if "message" in chunk and "content" in chunk["message"]:
            content = chunk["message"]["content"]
            accumulated_response += content
            print(content, end="", flush=True)
    print("\nProcessing complete response...")

    # Extract the CSV part from the response
    csv_text = re.findall(r"```(?:csv)?\n(.*?)```", accumulated_response, re.DOTALL)[0]
    # Convert CSV to list of dicts
    results = []
    for line in csv_text.strip().split('\n')[1:]:  # Skip header
        if ',' in line:
            key, value = line.split(',', 1)
            # Remove quotes if present
            value = value.strip('"')
            results.append({"key": key.strip(), "value": value.strip()})
    
    return results


async def extract_document_info(image: MatLike) -> list[dict]:
    """Main pipeline to extract document information"""
    df_results = perform_ocr(image)
    print("Running LLM...")
    processed_results = await proces_with_llm(df_results)
    processed_results = include_bboxes(df_results, processed_results)
    return processed_results


import cv2

if __name__ == "__main__":

    # Check data folder for available pics
    document_path = "c:\\Users\\jcsan\\Downloads\\Examples\\1Sty_3CL_7x9m_Standard_page-0001.jpg"
    image = cv2.imread(document_path)
    # image = cv2.resize(image, (0, 0), fx=1.2, fy=1.2)
    h = image.shape[0]
    w = image.shape[1]

    dpi = 200
    scale = 1

    if w < h:
        scale = dpi/(w/8.5)
    else:
        scale = dpi/(h/8.5)
    
    if scale != 1:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    image = preprocess._preprocess(preprocess.steps, image)
    # scale by 2
    result = asyncio.run(extract_document_info(image))
    print(result)
    preview(image, result, "./output.png")
