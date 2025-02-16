# Bounding box recovery: Labrador version

def include_bboxes(detailed_tess_results: list[dict], detected: list[dict]) -> list[dict]:
    """Include bounding boxes to processed results
    detailed: list of {text, bbox: [int,int,int,int]} dictionaries
    # bbox is unnormalized
    detected: list of {key, value} dictionaries, key is field name, value is field value
    """
    
    text_to_indexes: dict[str, int] = {}

    for index, result in enumerate(detailed_tess_results):
        text = result['text'].lower()
        
        if text in text_to_indexes:
            text_to_indexes[text].append(index)
        else:
            text_to_indexes[text] = [index]
    
    # List of {key, value, bbox} dictionaries
    results: list[dict] = []
    
    # Loop through detected fields and find the bounding box    
    for _item in detected:
        item = _item.copy()
        
        # Get the shortest path of nodes showing the text
        path = get_phrase_path(text_to_indexes, item['value'], item['key'])
        
        if path is not None:
            # Get the bbox of the path if found
            bbox = get_path_bbox(detailed_tess_results, path['route'])
            item['bbox'] = bbox
            results.append(item)
        else:
            print(f"Bounding box not found for {item['key']}")
    
    return results
    
    
    
def get_phrase_path(text_to_indexes: dict[str, int], phrase: str, key) -> dict:
    print(phrase)    
    words = phrase.split()
    x = len(words)
    
    skip_distance = 3 # distance to add when skipping a word in the phrase
    max_distance = x**1.5
    
    paths = [{'distance': 0, 'route': []}]
    
    for word in words:
        word = word.lower()
        
        new_paths = []
        
        if word not in text_to_indexes:
            for path in paths:
                distance = path['distance']
                route = path['route']
                
                if distance + skip_distance < max_distance:
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
                
                new_distance = 0
                if len(route) > 0:
                     new_distance = abs(route[-1] - index) - 1
                
                if new_distance <= skip_distance and distance + new_distance < max_distance:
                    new_paths.append({
                        'distance': distance + new_distance,
                        'route': route + [index]
                    })
                
                if skip_distance <= new_distance and distance + skip_distance < max_distance:
                    # Add new path that skips a word in the phrase instead
                    new_paths.append({
                        'distance': distance + skip_distance,
                        'route': route
                    })
                
        paths = new_paths
        
    paths = sorted(paths, key=lambda x: x['distance'])
    
    if len(words) <= 0 or len(paths) <= 0 or len(paths[0]['route']) <= 0:
        return None
    
    return paths[0]



def get_path_bbox(results: list[dict], route: list[int]) -> tuple[int, int, int, int]:
    bboxes = [results[index]['bbox'] for index in route]
    
    x0 = min(int(bbox[0]) for bbox in bboxes)
    y0 = min(int(bbox[1]) for bbox in bboxes)
    x1 = max(int(bbox[2]) for bbox in bboxes)
    y1 = max(int(bbox[3]) for bbox in bboxes)
    
    return (x0, y0, x1, y1)


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
