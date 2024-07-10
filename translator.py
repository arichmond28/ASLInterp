from sentence_transformers import SentenceTransformer, util
from PIL import Image
import numpy as np
import cv2 as cv
import os

def main():
    image_path = 'house1.jpg'
    
    # Verify if the file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return

    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Failed to load the image '{image_path}'.")
        return

    # Apply Gaussian blur to reduce noise
    blurred_img = cv.GaussianBlur(img, (5, 5), 0)

    # Use Canny edge detection for better edge detection
    edges = cv.Canny(blurred_img, 50, 150)

    # Find contours
    contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(img)
    
    for cnt in contours:
        cv.drawContours(mask, [cnt], -1, 255, thickness=-1)
    
    # Apply bitwise operations correctly
    isolated = cv.bitwise_and(img, img, mask=mask)
    mask_inv = cv.bitwise_not(mask)
    final_im = cv.bitwise_and(img, img, mask=mask_inv)
    final_im[final_im > 140] = 0

    # Save the processed image
    cv.imwrite('house1hands.jpg', final_im)

    # Draw contours on the original color image
    color_img = cv.imread(image_path)
    if color_img is None:
        print(f"Error: Failed to load the color image '{image_path}'.")
        return
    
    cv.drawContours(color_img, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

    # Save the image with contours
    cv.imwrite('house1_with_contours.jpg', color_img)

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

