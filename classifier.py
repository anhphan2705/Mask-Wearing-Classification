# Import necessary libraries and modules
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import time
import glob
import argparse

MODEL_DIRECTORY = './EfficientNet-Models/B0/9350/trained_model.pth'
FONT = "arial.ttf"
FONT_SIZE = 20
PRETRAIN_MODEL = 'efficientnet-b0'

use_gpu = torch.cuda.is_available()


def show_image(header, image):
    """
    Displays an image in a window with the specified header.
    
    Args:
        header (str): Header for the image window.
        image (numpy.ndarray): Image array to be displayed.
    """
    print("[Console] Showing image")
    cv2.imshow(header, image)
    cv2.waitKey()
    
    
def write_image(directory, image):
    """
    Saves an image to the specified directory.
    
    Args:
        directory (str): Path to the directory where the image will be saved.
        image (numpy.ndarray): Image array to be saved.
    """
    print("[Console] Saving image")
    cv2.imwrite(directory, image)


def get_images(directory):
    """
    Loads images from the specified directory.
    
    Args:
        directory (str): Path to the directory containing the images.
        
    Returns:
        List of loaded images (list of numpy.ndarray).
        
    Raises:
        Exception: If the directory is invalid or no images are found.
    """
    print("[Console] Accessing folder")
    image_paths = glob.glob(directory)
    print(image_paths)
    if len(image_paths) == 0:
        raise Exception("[INFO] Invalid directory")
    images = []
    # Add images to memory
    print("[Console] Loading Images")
    for image_path in image_paths:
        image = cv2.imread(image_path)
        images.append(image)
    print(f"[INFO] Loaded {len(images)} image(s)")
    return images


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
    for i, image in enumerate(images):
        image = Image.fromarray(image)
        image = image.resize((400, 400))
        I1 = ImageDraw.Draw(image)
        # I1.text((10, 10), f"{labels[index]} ({confs[index]:4f})", fill=(255, 0, 0), font=font_setting)                # with confidence index
        I1.text((10, 10), f"{labels[i]}", fill=(0, 255, 0), font=font_setting)                                      # without confidence index
        image = np.array(image)
        image_w_label.append(image)
        
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
        transforms.ToPILImage(),
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
            labels.append('no-mask')
            confs.append(round(float(conf.cpu()), 4))
        else:
            print('[INFO] Labeling went wrong')

    elapsed_time = time.time() - since

    return labels, confs, elapsed_time

if __name__ == "__main__":
    # Args Parser
    parser = argparse.ArgumentParser(prog='detect-mask',
                                     epilog='Text help'
                                     )
    parser.add_argument('-d', '--dir',
                        type=str,
                        default="./data/test/random/*.jpg",
                        help='Path of input images')
    parser.add_argument('-o', '--out',
                        type=str,
                        default="./output",
                        help='Path of input images')
    args = parser.parse_args()
    
    input_dir = args.dir
    output_dir = args.out
    
    images = get_images(input_dir)
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
    # Output API
    print('[INFO] Returning output')
    response_text = f'Label : {labels} with confidence {confs} in time {(elapsed_time // 60):.0f}m {(elapsed_time % 60):.0f}s'
    
    print(response_text)
    for i, image in enumerate(image_w_label):
        out = output_dir + f'/im_{i}.jpg'
        show_image("image", image)
        write_image(out, image)