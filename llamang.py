import asyncio
import re
import pandas as pd
import pytesseract
from ollama import AsyncClient
from cv2.typing import MatLike
from preview import preview

# from phrase_bbox import include_bboxes
from search import include_bboxes

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
    
    text = " ".join(df["text"].tolist())
    print("TEXT", text)

    async for chunk in await client.chat(
        # model="llama3.2:latest",
        model="deepseek-r1:latest",
        options={
            "temperature": 0.2, # configures the model's creativity
        },
        messages=[
            {
                "role": "system",
                "content": """
                YOUR TASK: Extract key information like form fields, or some questions from a document's text by identifying keywords and their corresponding values, returning them as CSV data.

                STEPS:
                1. __Text Preprocessing__: Clean and normalize the text for invalid control characters.
                2. __Structured Data Extraction__:
                    - Identify named entities and their relationships
                    - Flatten related fields so as to prevent nested structures
                    - Don't include named entities that are too long, try to limit to 5 words
                    - Recognize table-like structures and extract as new pairs
                3. __Final output__: 
                    - Format the output as CSV with two columns: key,value
                    - First line must be the header: key,value
                    - Properly escape commas and quotes in values
                    - Example format:
                        ```csv
                        key,value
                        name,John Smith
                        amount,42
                        address,"123 Main St, Suite 100"
                        ```
                    - Wrap the output in CSV code block: ```csv ... ```
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
    document_path = "./data/business_permit.jpg"
    image = cv2.imread(document_path)
    # scale by 2
    image = cv2.resize(image, (0, 0), fx=2, fy=2)
    result = asyncio.run(extract_document_info(image))
    print(result)
    preview(image, result, "./output.png")
