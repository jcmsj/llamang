from functools import reduce
from typing import Callable
from numpy import ndarray
import matplotlib.pyplot as plt
import cv2

def original(image: ndarray) -> ndarray:
    """Return the original image"""
    return image

def final(image: ndarray) -> ndarray:
    """Final preprocessing step"""
    return gray2bgr(image)

def gray2bgr(image: ndarray) -> ndarray:
    """Convert grayscale to BGR"""
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

def bgr2rgb(image: ndarray) -> ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb2gray(image: ndarray) -> ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
def individual_preprocess(steps:list[Callable], image: ndarray) -> dict[str, cv2.typing.MatLike]:
    d:dict[str,cv2.typing.MatLike] = {}
    for i, step in enumerate(steps):
        # get callable name
        name = step.__name__
        # get the image
        _image = step(image)
        # convert to rgb
        _image = bgr2rgb(_image)
        d[name] = _image
    return d
def transformations_preprocess(steps:list[Callable], image: ndarray) -> dict[str,cv2.typing.MatLike]:
    d:dict[str,cv2.typing.MatLike] = {}
    for i, step in enumerate(steps):
        # get callable name
        name = step.__name__
        # get the image
        image = step(image)
        d[f"{i}:{name}"] = image
    return d

def threshold(image: ndarray) -> ndarray:
    """Threshold an image"""
    # get optimal global threshold
    [optimal_treshold,_] = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return cv2.threshold(image, optimal_treshold, 255, cv2.THRESH_OTSU)[1]

def denoise(image: ndarray) -> ndarray:
    """Denoise an image"""
    return cv2.fastNlMeansDenoising(image, image, h=3, templateWindowSize=7, searchWindowSize=17)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
def dilate(image: ndarray) -> ndarray:
    """Dilate an image"""
    return cv2.dilate(image, kernel, iterations=1)

def erode(image: ndarray) -> ndarray:
    """Erode an image"""
    return cv2.erode(image, kernel, iterations=1)

def _preprocess(methods: list[Callable], image: ndarray) -> ndarray:
    """Preprocess an image"""
    return reduce(lambda img, method: method(img), methods, image)

def add_border(image: ndarray) -> ndarray:
    """Add borders to an image"""
    # Add borders, at least 10 px - https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#borders
    return cv2.copyMakeBorder(
        image, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )

def deskew(image: ndarray) -> ndarray:
    """Deskew an image"""
    # Deskew - https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#deskew
    raise NotImplementedError("Deskew not implemented")
    return image
def plot_preprocessed(d: dict[str, cv2.typing.MatLike]):
    import matplotlib.pyplot as plt

    cols = 2
    rows = (len(d) +cols -1) // cols
    # double the scale
    fig, axs = plt.subplots(rows, cols, figsize=(18, 10))
    for i, key in enumerate(d):
        ax = axs[i // cols, i % cols]
        ax.imshow(d[key], cmap="gray")
        ax.set_title(key)
        ax.axis("off")

# Steps represent the in-order preprocessing steps done
# steps = [original,rgb2gray,threshold, erode, denoise, add_border, final]
steps = [original,rgb2gray,denoise, add_border, final]
def main():
    """Run if main module"""
    # from args, get directory of images
    # preprocess images
    from argparse import ArgumentParser

    parser = ArgumentParser()
    # images_dir
    parser.add_argument("--dir", "-d", help="Directory of images to preprocess")
    # output_dir
    parser.add_argument(
        "--out-dir", "-od", help="Directory to save preprocessed images. Only works w/ --dir"
    )
    parser.add_argument("--file", "-f", help="File to preprocess")
    # Out
    parser.add_argument(
        "--output-file", "-of", help="Output directory for the prepreprocessed image. Only works w/ --file"
    )
    parser.add_argument(
        "--preview", "-p", help="Preview the preprocessing. Only works w/ --file",
        choices=["individual", "transformations","final"]
    )
    args = parser.parse_args()
    # print help
    if not args.dir and not args.file:
        parser.print_help()
        return


    if args.file:
        colored = cv2.imread(args.file, cv2.IMREAD_COLOR)
        colored = colored[..., ::-1]  # BGR -> RGB
        if args.preview == "individual":
            print("Previewing: individual")
            print("WARNING: Individual preview can't save")
            # prop 2nd
            steps.remove(rgb2gray)
            steps.pop()
            gray = rgb2gray(colored)
            d = individual_preprocess(steps,gray)
            # override original
            d["original"] = colored
            plot_preprocessed(d)
            plt.show()
            cv2.waitKey(0)
            return
        elif args.preview == "final":
            print("Previewing: final")
            image = _preprocess(steps,colored)
            plt.imshow(image)
            # disable axis
            plt.axis("off")
            plt.show()
        elif args.preview == "transformations":
            print("Previewing: transformations")
            p = transformations_preprocess(steps,colored)
            plot_preprocessed(p)
            plt.show()
        else:
            image = _preprocess(steps,colored)
        if args.output_file:
            cv2.imwrite(args.output_file, image)
            cv2.imshow("Preprocessed", image)
            return
        return
    
    # make the output directory if it doesn't exist
    import os

    os.makedirs(args.output_dir, exist_ok=True)
    # print that preview is ignored
    if args.preview:
        print("--preview flag is ignored when processing a directory")
    for root, _, files in os.walk(args.images_dir):
        for file in files:
            # print processing
            print(f"Processing {file}")
            image = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
            image = _preprocess(steps, image)
            output = os.path.join(args.output_dir, file)
            cv2.imwrite(output, image)
    print("Output saved to ", args.output_dir)
if __name__ == "__main__":
    main()
