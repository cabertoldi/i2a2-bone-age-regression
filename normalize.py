import cv2
import pandas as pd
import numpy as np
import os

def normalize_images(path):
    df = pd.read_csv(f'./data/{path}.csv')
    df = df.head(10)

    for filename in df['fileName']:
       _clean_image(path, filename)

def _clean_image(path, filename):
    image = cv2.imread(f"./data/images/{filename}")
    contours = _get_contours(image)
    
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (0, 255, 0), 2)
    (x, y, _) = np.where(mask == 255)
 
    if len(x) > 0 and len(y) > 0:
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))

        image = image[topx:bottomx+1, topy:bottomy+1]
        print('contours', len(contours))

        height, width, _ = image.shape
        print('width', width)

        if width > 800:
            # Cut the image in half
            width_cutoff = width // 2
            image = image[:, width_cutoff:]

    if not os.path.exists('./data/clean-images'):
        os.makedirs('./data/clean-images')

    cv2.imwrite(f"./data/clean-images/{filename}", image)

def _get_contours(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image_edges = cv2.Canny(image_gray, 40, 180)

    _, contours, hierarchy = cv2.findContours(image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours