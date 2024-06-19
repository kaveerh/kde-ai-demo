import base64
import boto3
import json
from PIL import Image
from random import randint
import io
import os

def generate_image(img_path, prompt):
    # Load the AWS Bedrock Runtime client
    bedrock_runtime = boto3.client("bedrock-runtime")

    # Load the input image from disk
    with open(img_path, "rb") as image_file:
        input_image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

    # Configure the inference parameters
    inference_params = {
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "text": prompt,
            "negativeText": "bad quality, low res",
            "image": input_image_base64,
            "maskPrompt": "body of pipeline"
        },
        "imageGenerationConfig": {
            "numberOfImages": 3,
            "quality": "premium",
            "height": 1024,
            "width": 1024,
            "cfgScale": 8.0,
            "seed": randint(0, 100000)
        }
    }

    # Invoke the model
    response = bedrock_runtime.invoke_model(
        modelId="amazon.titan-image-generator-v1",
        body=json.dumps(inference_params)
    )

    # Process the response
    response_body = json.loads(response["body"].read())
    images = response_body["images"]

    # Save the generated images to disk
    for i, image_data in enumerate(images):
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        output_file_name = f"output-{i + 1}.png"
        image.save(output_file_name)
