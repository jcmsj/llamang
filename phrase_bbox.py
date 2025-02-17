# Bounding box recovery: Labrador version
import pandas as pd

def include_bboxes(ocr_results: pd.DataFrame, llm_metadata: list[dict]) -> list[dict]:
    """Include bounding boxes to processed results
    ocr_results: DataFrame with columns [text, x1, y1, x2, y2]
    llm_metadata: list of {key, value} dictionaries
    Returns list of dictionaries with added bbox
    """

    ocr_tuples = []
    prev_row = None

    for _, row in ocr_results.iterrows():
        line_height = row["y2"] - row["y1"]

        if prev_row is not None:
            if prev_row["y2"] < row["y1"]:
                ocr_tuples.append((
                    "\n",
                    row["x1"],
                    row["y1"],
                    row["x1"],
                    row["y2"]
                ))
            elif row["x1"] - prev_row["x2"] > line_height*1.5:
                ocr_tuples.append((
                    "\t",
                    row["x1"],
                    row["y1"],
                    row["x1"],
                    row["y2"]
                ))

        
        ocr_tuples.append((
            row["text"],
            row["x1"],
            row["y1"],
            row["x2"],
            row ["y2"]
        ))

        prev_row = row
    
    text_to_indexes: dict[str, list[int]] = {}

    for index, result in enumerate(ocr_tuples):
        text = result[0].lower()
        
        if text in text_to_indexes:
            text_to_indexes[text].append(index)
        else:
            text_to_indexes[text] = [index]
    
    # List of {key, value, bbox} dictionaries
    results: list[dict] = []
    
    # Loop through detected fields and find the bounding box    
    for _item in llm_metadata:
        item = _item.copy()
        
        # Get the shortest path of nodes showing the text
        path = get_phrase_path(text_to_indexes, item['value'], item['key'])
        
        if path is not None:
            # Get the bbox of the path if found
            bbox = get_path_bbox(ocr_tuples, path['route'])
            item['bbox'] = bbox
            results.append(item)
        else:
            print(f"Bounding box not found for {item['key']}")
    
    return results
    
    
    
def get_phrase_path(text_to_indexes: dict[str, list[int]], phrase: str, key) -> dict:
    import math
    print(phrase)    
    words = phrase.split()
    x = len(words)

    if x <= 0:
        return None
    
    skip_distance = 3 # distance to add when skipping a word in the phrase
    max_distance = x*math.log(x)
    
    paths = [{'distance': 0, 'route': [-1]}]

    if "\n" in text_to_indexes:
        paths.extend([{'distance': 0, 'route': [i]} for i in text_to_indexes["\n"]])
    if "\t" in text_to_indexes:
        paths.extend([{'distance': 0, 'route': [i]} for i in text_to_indexes["\t"]])
    
    
    for word in words:
        word = word.lower()
        
        new_paths = []
        
        if word not in text_to_indexes:
            for path in paths:
                distance = path['distance']
                route = path['route']
                
                if distance + skip_distance <= max_distance:
                    new_paths.append({
                        'distance': distance + skip_distance,
                        'route': route
                    })
                
            paths = new_paths
            continue
            
        indexes = text_to_indexes[word]
        
        for path in paths:
            for index in indexes:
            
                distance = path['distance']
                route = path['route']
                
                if index in route:
                    continue
                
                new_distance = index - route[-1] - 1

                if new_distance < 0:
                    continue
                
                if new_distance <= skip_distance and distance + new_distance <= max_distance:
                    new_paths.append({
                        'distance': distance + new_distance,
                        'route': route + [index]
                    })
                
                if skip_distance <= new_distance and distance + skip_distance <= max_distance:
                    # Add new path that skips a word in the phrase instead
                    new_paths.append({
                        'distance': distance + skip_distance,
                        'route': route
                    })
                
        paths = new_paths
        
    paths = sorted(paths, key=lambda x: x['distance'])
    
    if len(paths) <= 0 or len(paths[0]['route']) <= 0:
        return None
    
    return paths[0]



def get_path_bbox(results: list[tuple], route: list[int]) -> tuple[int, int, int, int]:
    bboxes = [results[index] for index in route]
    
    x1 = min(int(bbox[1]) for bbox in bboxes)
    y1 = min(int(bbox[2]) for bbox in bboxes)
    x2 = max(int(bbox[3]) for bbox in bboxes)
    y2 = max(int(bbox[4]) for bbox in bboxes)
    
    return (x1, y1, x2, y2)


def main():
    import json

    with open('llm_output.json', 'r') as file:
        detected = json.load(file)

    with open('ocr_results.json', 'r') as file:
        detailed_tess_results = json.load(file)
        
    results = include_bboxes(detailed_tess_results, detected)

    print(results)


if __name__ == '__main__':
    main()
