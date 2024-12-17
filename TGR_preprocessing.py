import os
import shutil
from PIL import Image
from torchvision import transforms
import torch
from tqdm import *
from PIL import Image
import TGR_method
from TGR_method import *


# Define preprocessing function using torchvision.transforms
def preprocess_image(image_path, transform):
    """
    Load and preprocess a single image.
    Args:
        image_path: Path to the image file.
        transform: torchvision.transforms object for preprocessing.
    Returns:
        Tensor image after preprocessing.
    """
    image = Image.open(image_path).convert('RGB')  # Ensure it's RGB
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze_(0)
    return image_tensor

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
# Define the image preprocessing pipeline
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        #transforms.Resize([112, 112]),
        transforms.ToTensor(),
        normalize,
    ])

# Function to process folders and save preprocessed images
def process_folders(input_root, output_root, transform):
    """
    Preprocess images in all folders under input_root and save to output_root.
    Args:
        input_root: Path to the root folder containing input folders.
        output_root: Path to the root folder to save preprocessed images.
        transform: torchvision.transforms for image preprocessing.
    """
    # Ensure output directory exists
    os.makedirs(output_root, exist_ok=True)
    counter = 0
    target = torch.ones((1), dtype=torch.long).cuda()
    # Traverse each folder in the input_root directory
    for folder_name in os.listdir(input_root):
        input_folder = os.path.join(input_root, folder_name)
        output_folder = os.path.join(output_root, folder_name)

        # Ensure input is a folder
        if not os.path.isdir(input_folder):
            continue

        # Create a corresponding output folder
        os.makedirs(output_folder, exist_ok=True)
        TGR_attack = TGR(model_name='vit_base_patch16_224')
        real_target = target * counter
        # Process each image in the folder
        for img_name in os.listdir(input_folder):
            input_image_path = os.path.join(input_folder, img_name)
            output_image_path = os.path.join(output_folder, img_name)

            try:
                # Preprocess image
                image_tensor = preprocess_image(input_image_path, transform)
                adv_images = TGR_attack(image_tensor, real_target)

                # Convert back to PIL Image for saving
                adv_images = adv_images.detach()
                adv_images = adv_images.cpu().squeeze(0)
                image_pil = transforms.ToPILImage()(adv_images)
                image_pil.save(output_image_path)

                print(f"Processed: {input_image_path} -> {output_image_path}")
            except Exception as e:
                print(f"Error processing {input_image_path}: {e}")

        print(torch.cuda.memory_summary())
# Paths: Update these paths accordingly
input_root = "/home/xiaofengzheng/Desktop/jliu206/val/data/jiayang/imagenet/val"  # Replace with your input root folder
output_root = "/home/xiaofengzheng/Desktop/jliu206/val/data/jiayang/imagenet/attack_vals"  # Replace with your desired output root folder

# Run preprocessing
process_folders(input_root, output_root, transform)
