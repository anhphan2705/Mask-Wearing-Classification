# Import necessary libraries and modules
from fastapi import FastAPI, UploadFile, Response
from fastapi.responses import HTMLResponse
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import time

app = FastAPI()
MODEL_DIRECTORY = './EfficientNet-Models/B0/9350/trained_model.pth'
FONT = "arial.ttf"
FONT_SIZE = 25
PRETRAIN_MODEL = 'efficientnet-b0'

use_gpu = torch.cuda.is_available()

def convert_byte_to_arr(byte_image):
    """
    Convert an image in byte format to a PIL Image object (RGB format).

    Args:
        byte_image (bytes): The image data in byte format.

    Returns:
        Image.Image: A PIL Image object representing the image in RGB format.
    """
    arr_image = Image.open(BytesIO(byte_image)).convert("RGB")
    return arr_image


def convert_arr_to_byte(arr_image):
    """
    Convert a numpy array image (RGB format) to byte format (JPEG).

    Args:
        arr_image (numpy.ndarray): The image data as a numpy array in RGB format.

    Returns:
        bytes: The image data in byte format (JPEG).
    """
    arr_image = np.array(arr_image)
    arr_image = cv2.cvtColor(arr_image, cv2.COLOR_RGB2BGR)
    # Encode the image as JPEG format
    success, byte_image = cv2.imencode(".jpg", arr_image)
    if success:
        return byte_image.tobytes()
    else:
        raise Exception("Cannot convert array image to byte image")
    

def multiple_to_one(images):
    """
    Combine multiple images horizontally into a single image.

    Args:
        images (List[Image.Image]): List of PIL Image objects representing the input images.

    Returns:
        Image.Image: A new PIL Image object containing the input images combined horizontally.
    """
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    return new_im


def assign_image_label(images, labels, confs, font="arial.ttf", font_size=25):
    """
    Add labels to the input images.

    Args:
        images (List[Image.Image]): List of PIL Image objects representing the input images.
        labels (List[str]): List of labels corresponding to the input images.
        confs (List[float]): List of confidence level of each prediction for the corresponding input image
        font (str, optional): The font file to be used for the labels. Defaults to "arial.ttf".
        font_size (int, optional): The font size for the labels. Defaults to 25.

    Returns:
        List[Image.Image]: List of PIL Image objects with labels added to the top left corner.
    """
    image_w_label = []
    font_setting = ImageFont.truetype(font, font_size)
    for index in range(len(images)):
        I1 = ImageDraw.Draw(images[index])
        # I1.text((10, 10), f"{labels[index]} ({confs[index]:4f})", fill=(255, 0, 0), font=font_setting)                # with confidence index
        I1.text((10, 10), f"{labels[index]}", fill=(255, 0, 0), font=font_setting)                                      # without confidence index
        image_w_label.append(images[index])
        
    return image_w_label
    

def get_data(np_images):
    """
    Prepare the list of numpy array images for classification.

    Args:
        np_images (List[numpy.ndarray]): List of numpy array images (RGB format).

    Returns:
        List[torch.Tensor]: List of preprocessed images as PyTorch tensors.
    """
    data_transform = transforms.Compose([
        transforms.Resize(254),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    data = []
    for image in np_images:
        # Convert numpy ndarray [3, 224, 224] to PyTorch tensor
        image = data_transform(image)
        # Expand to [batch_size, 3, 224, 224]
        image = torch.unsqueeze(image, 0)
        data.append(image)
    return data


def get_pretrained_model(model_dir=MODEL_DIRECTORY, weights=PRETRAIN_MODEL):
    """
    Retrieve the VGG-16 pre-trained model and modify the classifier with a fine-tuned one.

    Args:
        model_dir (str, optional): Directory path for loading a pre-trained model state dictionary. Defaults to ''.
        weights (str or dict, optional): Pre-trained model weights. Defaults to models.vgg16_bn(pretrained=True).state_dict().

    Returns:
        torchvision.models.vgg16_bn: VGG-16 model with modified classifier.
    """
    print("[INFO] Getting VGG-16 pre-trained model...")
    # Load pretrained model
    model = EfficientNet.from_pretrained(weights)
    model.eval()
    # Freeze training for all layers
    for param in model.parameters():
        param.requires_grad = False
    # Get number of features in the _fc layer
    num_features = model._fc.in_features
    # Add custom layer with custom number of output classes
    model._fc = nn.Linear(num_features, 2)
    # Load VGG-16 pretrained model
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    print("[INFO] Loaded VGG-16 pre-trained model\n", model, "\n")

    return model


def get_prediction(model, images):
    """
    Perform image classification using the provided model.

    Args:
        model (torchvision.models.vgg16_bn): The fine-tuned VGG-16 model.
        images (List[torch.Tensor]): List of preprocessed images as PyTorch tensors.

    Returns:
        Tuple[List[str], List[float], float]: A tuple containing the list of predicted labels, the confidence for the predictions, and the time taken for classification.
    """
    since = time.time()
    labels = []
    confs = []
    model.train(False)
    model.eval()

    for image in images:
        with torch.no_grad():
            if use_gpu:
                image = Variable(image.cuda())
            else:
                image = Variable(image)

        outputs = model(image)
        
        probs = torch.nn.functional.softmax(outputs.data, dim=1)
        conf, pred = torch.max(probs, 1)
        
        if pred == 0:
            labels.append('mask')
            confs.append(round(float(conf.cpu()), 4))
        elif pred == 1:
            labels.append('no_mask')
            confs.append(round(float(conf.cpu()), 4))
        else:
            print('[INFO] Labeling went wrong')

    elapsed_time = time.time() - since

    return labels, confs, elapsed_time


@app.get("/")
def welcome_page():
    """
    Serves the root route ("/") and displays a welcome message with a link to the API documentation.

    Returns:
        fastapi.responses.HTMLResponse: HTML response with a welcome message and a link to the API documentation.
    """
    return HTMLResponse(
        """
        <h1>Welcome to Banana</h1>
        <p>Click the button below to go to /docs/:</p>
        <form action="/docs" method="get">
            <button type="submit">Visit Website</button>
        </form>
    """
    )


@app.post("/mask-wearing-classification")
async def dog_cat_classification(in_images: list[UploadFile]):
    """
    API endpoint to classify multiple images as either wearning mask or not using a fine-tuned EfficientNet-B0 model.

    Args:
        in_images (List[UploadFile]): List of images in JPG format to be classified.

    Returns:
        fastapi.responses.Response: Images with a label on the top left corner as a response.
    """
    images = []
    for in_image in in_images:
        byte_image = await in_image.read()
        arr_image = convert_byte_to_arr(byte_image)
        images.append(arr_image)
    print(f'[INFO] Received {len(images)} images')
    
    # Preparing data and loading the model
    data = get_data(images)
    model = get_pretrained_model()
    
    # Use GPU if available
    print('[INFO] Classification in progress')
    print("[INFO] Using CUDA") if use_gpu else print("[INFO] Using CPU")
    if use_gpu:
        torch.cuda.empty_cache()
        model.cuda()
    
    labels, confs, elapsed_time = get_prediction(model, data)
    print(f"[INFO] Label : {labels} with confidence {confs} in time {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s")

    # Add label and confidence level to the top left corner of the input image
    print('[INFO] Writing labels onto output images')
    image_w_label = assign_image_label(images, labels, confs, font=FONT, font_size=FONT_SIZE)
    # Combined multiple images into one
    image_combined = multiple_to_one(image_w_label)
    # Output API
    print('[INFO] Returning output')
    byte_images = convert_arr_to_byte(image_combined)
    response_text = f'Label : {labels} with confidence {confs} in time {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s'
    response = Response(content=byte_images, media_type="image/jpg")
    response.headers["Result"] = response_text
    
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
