import time
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import cv2

# Function to remove watermark using inpainting (OpenCV)
def remove_watermark(img):
    # Convert the image to OpenCV format (BGR)
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Define the watermark area to mask (You can adjust the area based on where the watermark is located)
    # Example: watermark is in the bottom right corner
    height, width, _ = img_cv.shape
    x, y, w, h = int(width * 0.8), int(height * 0.9), int(width * 0.2), int(height * 0.1)  # Bottom-right corner

    # Create a mask for the watermark area
    mask = np.zeros(img_cv.shape, dtype=np.uint8)
    mask[y:y+h, x:x+w] = 255  # Set the region of the watermark to white

    # Inpaint the image using the mask
    result = cv2.inpaint(img_cv, cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY), inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Convert back to PIL format
    result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    return result_img

# Function to convert the image to a tensor, store it in a variable, and visualize from the tensor
def process_and_visualize_image(prompt, width=768, height=768, model='flux'):
    url = f"https://image.pollinations.ai/prompt/{prompt}?width={width}&height={height}&model={model}&seed={42}"
    response = requests.get(url)

    if response.status_code == 200:
        # Open the image from the response
        img = Image.open(BytesIO(response.content))
        
        # Remove watermark (using the inpainting function)
        img_without_watermark = remove_watermark(img)
        
        # Convert the image to a tensor
        transform = transforms.ToTensor()
        img_tensor = transform(img_without_watermark)

        # Store the tensor in a variable (no file saving)
        tensor_variable = img_tensor

        # Convert the tensor back to an image
        loaded_img = transforms.ToPILImage()(tensor_variable)

        return loaded_img
    else:
        st.error(f"Failed to generate image. Status code: {response.status_code}")
        return None

# Streamlit UI for input
st.title("Image Generation with Prompt")
prompt = st.text_input("Enter a prompt and press Enter:")
width = st.slider("Select width", min_value=100, max_value=1920, value=1280)
height = st.slider("Select height", min_value=100, max_value=1080, value=720)

# Create a placeholder for live clock
live_clock = st.empty()

# Button to trigger image generation
if st.button("Generate Image"):
    start_time = time.time()  # Record the start time

    with st.spinner("Generating image... Please wait."):

        # Generate and display image once the process is complete
        processed_image = process_and_visualize_image(prompt, width, height)
        if processed_image:
            st.image(processed_image, use_container_width=True)

    # Display total elapsed time after image is generated
    total_time = time.time() - start_time
    st.write(f"Image generated and displayed in {total_time:.2f} seconds.")
