from dotenv import find_dotenv, load_dotenv
from transformers import pipeline

# img2text
def img2text(url):
    imageToText = pipeline('image-to-text', model="Salesforce/blip-image-captioning-base")

    text = imageToText(url)[0]["generated_text"]
    print(text) # a man and woman dancing on a dance floor

    return text

# LLM



# text to speech



load_dotenv(find_dotenv())

img2text("photo_01.png")

