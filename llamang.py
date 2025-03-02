import argparse
import asyncio
import json
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
            results.append(
                {
                    "text": detailed_data["text"][i],
                    "x1": detailed_data["left"][i],
                    "y1": detailed_data["top"][i],
                    "x2": detailed_data["left"][i] + detailed_data["width"][i],
                    "y2": detailed_data["top"][i] + detailed_data["height"][i],
                }
            )

    return pd.DataFrame(results)


async def proces_with_llm(df: pd.DataFrame, stream=False) -> list[dict]:
    """Process OCR results with deepseek model"""
 

    # build the full text input, add spaces, but also try to restore the newlines, by comparing the previous text's bbox w/ the current text's bbox, if they do not overlap, add a newline
    text = ""
    for i, row in df.iterrows():
        if i == 0:
            text += row["text"]
        else:
            prev_row = df.iloc[i - 1]  # type: ignore
            line_height = row["y2"] - row["y1"]
            if prev_row["y2"] < row["y1"]:
                text += "\n" + row["text"]
            elif row["x1"] - prev_row["x2"] > line_height * 1.5:
                text += "\t" + row["text"]
            else:
                text += " " + row["text"]

    # print("TEXT", text)
    # OLLAMA = "llama3.2:latest"
    # OLLAMA="llama3.2:3b-instruct-q4_1"
    OLLAMA="llama3.2:3b-instruct-q6_K"
    # OLLAMA="llama3.2:3b-instruct-q4_K_M" # Instruct models are for doing specific tasks, K-M provides a good balance of quality and speed
    # OLLAMA = "qwen2.5-coder:7b-instruct-q4_K_M"
    model = OLLAMA
    instruction_prompt_role = "system" if model == OLLAMA else "user"
    messages = [
            {
                "role": instruction_prompt_role,
                "content": f"""
                ROLE: You are an expert document layout analyzer. 
                TASK: Given the `ocr-input` variable text extracted from a document region, provide a list of keyword:value pairs.

                GUIDE:
                * __Keyword__: a word or short phrase that acts as a label, indicating the type of information that follows.
                    * Examples: name, address, date, amount, total, etc.
                    * hinted by its delimiter (such as colon or other punctuations), position (e.g. table headers), or context.
                    * Prefer words or short phrases less than 5 words.
                    * preserve original OCR output for easy retrieval later. E.g. don't change `Full name` into `full_name`.
                    * try to include metadata as keys, such as title of the document, dates, etc.
                    * consider spatial layout, such as proximity to other keywords or values.
                    * it is fine to have keywords not explicitly present in the text, but are implied by the context.
                * __Value__: the actual information associated with the keyword.
                    * Could be named entities, dates, numbers, or other text.
                    * prefer shorter strings.
                    * keep as string, dont make into an object or array.
                * Always flatten nested objects.
                Output: Identify at least 15 of these pairs as json.
                Example:
                ```json
                {{'Full Name': 'John Doe', 'address': '123 Main St', 'date': '2022-01-01'}}
                ```
                """,
            },
            {"role": "user", "content": f"<ocr-input>{text}</ocr-input>"},
        ]
    client = AsyncClient()
    accumulated_response = ""
    if stream:
        async for chunk in await client.chat(
            model=model,
            # model="llama3.2:1b",
            # model="qwen2.5:3b",
            # model="deepseek-r1:1.5b",
            # model="deepseek-r1:latest", # prompt is still incorrect with this model
            options={
                # "temperature": 0.6, # configures the model's creativity
                "temperature": 0.2,
            },
            format="json",
            messages=messages,
            stream=True,
        ):
            if "message" in chunk and "content" in chunk["message"]:
                content = chunk["message"]["content"]
                accumulated_response += content
                print(content, end="", flush=True)
    else:
        response = await client.chat(
            model=model,
            # model="llama3.2:1b",
            # model="qwen2.5:3b",
            # model="deepseek-r1:1.5b",
            # model="deepseek-r1:latest", # prompt is still incorrect with this model
            options={
                # "temperature": 0.6, # configures the model's creativity
                "temperature": 0.2,
            },
            format="json",
            messages=messages,
        )
        accumulated_response = response['message']['content']

    print("\nProcessing complete response...")

    # Extract the CSV part from the response
    json_response = json.loads(accumulated_response)  # { key: value }
    # now expand to
    # {key: actual key, value: actual value}
    results = []

    for key, value in json_response.items():
        # flatten nested objects
        if isinstance(value, dict):
            for k, v in value.items():
                results.append({"key": k, "value": v})
        elif isinstance(value, list):
            # skip
            pass
        else:
            results.append({"key": key, "value": value})

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
    args = argparse.ArgumentParser()
    args.add_argument(
        "--document", type=str, required=True, help="Path to document image"
    )
    # allow autoscaling, default true via an argument
    args.add_argument(
        "--autoscale", action="store_true", help="Autoscale image to fit 8.5x11"
    )
    args = args.parse_args()
    image = cv2.imread(args.document)
    # image = cv2.resize(image, (0, 0), fx=1.2, fy=1.2)
    h = image.shape[0]
    w = image.shape[1]

    if args.autoscale:
        print("Autoscaling image...")
        dpi = 200
        scale = 1

        if w < h:
            scale = dpi / (w / 8.5)
        else:
            scale = dpi / (h / 8.5)

        if scale != 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    image = preprocess._preprocess(preprocess.steps, image)
    # scale by 2
    result = asyncio.run(extract_document_info(image))
    print(result)
    preview(image, result, "./output.png")
