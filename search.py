# Bounding box recovery: San Juan version

from difflib import SequenceMatcher
from typing import Optional, Tuple, Callable
import re
import pandas as pd

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return re.sub(r'\s+', ' ', str(text).lower().strip())

def try_exact_match(answer: str, words: list[str], details: list[tuple[str, int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Strategy 1: Exact match"""
    answer_words = answer.split()
    for i in range(len(words)):
        for length in range(len(answer_words), 0, -1):
            window = " ".join(words[i : i + length])
            if window == answer:
                matched_boxes = details[i : i + length]
                return get_bbox_from_matches(matched_boxes)
    return None

def try_fuzzy_match(answer: str, words: list[str], details: list[tuple[str, int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Strategy 2: Fuzzy matching"""
    best_ratio = 0
    best_match = None
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            window = " ".join(words[i:j])
            ratio = SequenceMatcher(None, window, answer).ratio()
            if ratio > best_ratio and ratio > 0.8:
                best_ratio = ratio
                best_match = (i, j)
    
    if best_match:
        i, j = best_match
        return get_bbox_from_matches(details[i: j])
    return None

def try_token_match(answer: str, words: list[str], details: list[tuple[str, int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    """Strategy 3: Token-based matching"""
    answer_tokens = set(answer.split())
    for i in range(len(words)):
        window_tokens = set()
        for j in range(i, len(words)):
            window_tokens.update(words[j].split())
            if len(answer_tokens & window_tokens) / (len(answer_tokens )+0.0001) > 0.8:
                return get_bbox_from_matches(details[i:j+1])
    return None

def get_bbox_from_matches(matched_boxes: list[tuple[str, int, int, int, int]]) -> Tuple[int, int, int, int]:
    """Helper to get bounding box from matched boxes"""
    return (
        min(int(box[1]) for box in matched_boxes),  # x1
        min(int(box[2]) for box in matched_boxes),  # y1
        max(int(box[3]) for box in matched_boxes),  # x2
        max(int(box[4]) for box in matched_boxes)   # y2
    )

def find_text_span(answer: str, details: pd.DataFrame) -> Optional[tuple[int, int, int, int]]:
    """Find the bounding box using multiple strategies in sequence"""
    normalized_answer = normalize_text(answer)
    words = [normalize_text(text) for text in details["text"]]

    # Convert DataFrame rows to list of tuples for compatibility with existing functions
    details_tuples = [(row["text"], row["x1"], row["y1"], row["x2"], row["y2"]) 
                      for _, row in details.iterrows()]

    strategies: list[tuple[str, Callable]] = [
        ("exact", try_exact_match),
        ("fuzzy", try_fuzzy_match),
        # ("token", try_token_match),
    ]

    for strategy_name, strategy_fn in strategies:
        result = strategy_fn(normalized_answer, words, details_tuples)
        if result:
            print(f"Found match using {strategy_name} strategy for: {answer[:30]}...")
            return result
            
    return None

def include_bboxes(detailed_df: pd.DataFrame, process_results: list[dict]) -> list[dict]:
    """Include bounding boxes to processed results
    detailed_df: DataFrame with columns [text, x1, y1, x2, y2]
    process_results: list of {key, value} dictionaries
    Returns list of dictionaries with added bbox
    """
    results_with_bbox = []
    
    for item in process_results:
        field = item.copy()
        if "value" in field:
            bbox = find_text_span(field["value"], detailed_df)
            if bbox:
                field["bbox"] = bbox
                results_with_bbox.append(field)
            else:
                print(f"Bounding box not found for field {field['key']}")

    return results_with_bbox

def main():
    '''Run if main module'''
    # OUTDATED
    import json
    ocr_results_path = r"D:\code\kata\ocr_results.json"
    llm_result_path = r"D:\code\kata\llm_output.json"
    ocr_results = pd.read_json(ocr_results_path)

    llm_results = json.load(open(llm_result_path))
    print(llm_results)
    fields_with_bbox = include_bboxes(ocr_results, llm_results)
    print("Fields:", fields_with_bbox)

if __name__ == '__main__':
    main()
