from sentence_transformers import SentenceTransformer, util
from PIL import Image
import streamlit as st

def main():
  #Load CLIP model
  model = SentenceTransformer('clip-ViT-L-14')

  #Encode an image:
  img_emb = model.encode(Image.open('cookie1.jpg'))

  #Encode text descriptions
  text_emb = model.encode(Image.open('cookie1again.jpg'))

  img_emb2 = model.encode(Image.open('cookie2.jpeg'))

  #Compute cosine similarities 
  cos_scores = util.cos_sim(img_emb, text_emb)
  cos_scores2 = util.cos_sim(img_emb, img_emb2)
  print(cos_scores)
  print(cos_scores2)

main()