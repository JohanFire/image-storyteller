from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os

# img2text
def img2text(url):
    imageToText = pipeline('image-to-text', model="Salesforce/blip-image-captioning-base")

    text = imageToText(url)[0]["generated_text"]
    print(text) # a man and woman dancing on a dance floor

    return text

# LLM
def generate_story(scenario):
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;

    CONTEXT: {scenario}
    STORY:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True
    )

    story = story_llm.predict(scenario=scenario)
    print(story)

    return story


# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_HUB_API_TOKEN}"}
    payload = {
	"inputs": message,
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)
    # return response.json()


load_dotenv(find_dotenv())
HUGGINGFACE_HUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

scenario =  img2text("photo_01.png")
story = generate_story(scenario)
text2speech(story)
