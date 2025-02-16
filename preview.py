import cv2
from cv2.typing import MatLike

def preview(image:MatLike, results:list[dict], out_filename:str):

    for item in results:
        print(item)
        # if there's no bbox, skip
        if "bbox" not in item:
            continue
        x1, y1, x2, y2 = item["bbox"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            item["key"],
            (x1, y1),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
    # save
    cv2.imwrite(out_filename, image)

def main():
    '''Run if main module'''

if __name__ == '__main__':
    main()
