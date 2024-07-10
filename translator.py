from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import cv2 as cv
import os

def main():
    image_path = 'house1.jpeg'
    
    # Verify if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to load the image '{image_path}'.")
        return

    ret, thresh = cv.threshold(img, 9, 10, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    
    for cnt in contours:
        cv.drawContours(mask, [cnt], -1, 255, thickness=-1)
    
    mask_inv = 255 - mask
    final_im = mask_inv * img
    final_im[final_im > 140] = 0
    cv.imwrite('house1hands.jpg', final_im)

    # Load CLIP model
    model = SentenceTransformer('clip-ViT-L-14')

    # Encode an image:
    img_emb = model.encode(Image.open('house1.jpg'))

    # Encode text descriptions
    text_emb = model.encode(Image.open('house1.jpeg'))

    img_emb2 = model.encode(Image.open('house1.jpeg'))

    # Compute cosine similarities 
    cos_scores = util.cos_sim(img_emb, text_emb)
    cos_scores2 = util.cos_sim(img_emb, img_emb2)
    print(cos_scores)
    print(cos_scores2)

if __name__ == "__main__":
    main()
