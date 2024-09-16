import json
import boto3
import streamlit as st
import dotenv
from PIL import Image
from io import BytesIO
import base64
import io

dotenv.load_dotenv()

st.title("Amazon Bedrock Multimodal: Claude & SDXL 1.0")

REGION = "us-west-2"
client = boto3.client(service_name="bedrock-runtime", region_name=REGION)

# Model IDs
claude_model_id = "anthropic.claude-3-haiku-20240307-v1:0"  # text-to-text
sdxl_model_id = "stability.stable-diffusion-xl-v1"  # text-to-image

sd_presets = [
    "None",
    "3d-model",
    "analog-film",
    "anime",
    "cinematic",
    "comic-book",
    "digital-art",
    "enhance",
    "fantasy-art",
    "isometric",
    "line-art",
    "low-poly",
    "modeling-compound",
    "neon-punk",
    "origami",
    "photographic",
    "pixel-art",
    "tile-texture",
]

# Parsing functions for Claude (text) and SDXL (image)
def parse_text_stream(stream):
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            message = json.loads(chunk.get("bytes").decode())
            if message['type'] == "content_block_delta":
                yield message['delta']['text'] or ""
            elif message['type'] == "message_stop":
                return "\n"

def parse_image_response(stream):
    image_data = b""
    for event in stream:
        chunk = event.get('chunk')
        if chunk:
            message = json.loads(chunk.get("bytes").decode())
            if message['type'] == "image_data":
                image_data += message['image_bytes']
            elif message['type'] == "message_stop":
                break
    return image_data

# Convert base64 string to image
def convert_base64_to_image(base64_string):
    img_data = base64.b64decode(base64_string)
    img = Image.open(io.BytesIO(img_data))
    return img

# Generate image using Stable Diffusion
def generate_image_sd(bedrock_client, text, style):
    body = {
        "text_prompts": [{"text": text}],
        "cfg_scale": 10,
        "seed": 0,
        "steps": 50,
        "style_preset": style,
    }
    if style == "None":
        del body["style_preset"]
    body = json.dumps(body)
    response = bedrock_client.invoke_model(
        body=body, modelId=sdxl_model_id, accept="application/json", contentType="application/json"
    )
    response_body = json.loads(response.get("body").read())
    return response_body.get("artifacts")[0].get("base64")

# Main application interface
model_type = st.selectbox("Choose model type", ["Text-to-Text (Claude)", "Text-to-Image (SDXL 1.0)"])

prompt = st.text_input("Enter your prompt")

if prompt:
    if model_type == "Text-to-Text (Claude)":
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}],
                }
            ],
        })
        streaming_response = client.invoke_model_with_response_stream(
            modelId=claude_model_id, body=body,
        )
        st.subheader("Text Output Stream")
        stream = streaming_response.get("body")
        st.write("".join(parse_text_stream(stream)))

    elif model_type == "Text-to-Image (SDXL 1.0)":
        style = st.selectbox("Select image style", sd_presets)
        if st.button("Generate Image"):
            image_base64 = generate_image_sd(client, prompt, style)
            if image_base64:
                img = convert_base64_to_image(image_base64)
                st.image(img)