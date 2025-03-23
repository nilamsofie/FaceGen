import sys
import os
import numpy as np
import torch
import subprocess
import pickle
from pathlib import Path
from PIL import Image
#from ganspace.pca import PAC

# Ensure StyleGAN3 can be found
STYLEGAN3_DIR = "stylegan3"
sys.path.append(STYLEGAN3_DIR)

# Paths
STYLEGAN3_DIR = "stylegan3"
INTERFACEGAN_DIR = "interfacegan"
MODEL_PATH = "stylegan3/models/stylegan3-r-ffhq-1024x1024.pkl"  
OUTPUT_DIR = "outputs"
#Paths for boundaries
AGE_PATH = "boundaries/stylegan_ffhq_age_c_gender_boundary.npy"
POSE_PATH = "boundaries/stylegan_ffhq_pose_w_boundary.npy"
POSE2_PATH = "boundaries/stylegan_ffhq_pose_boundary.npy"
SMILE_PATH = "boundaries/stylegan_ffhq_smile_w_boundary.npy"
#Paths from ganspace
RGB_PATH = "ganspace/notebooks/data/steerability/stylegan_ffhq/ffhq_rgb_0.npy"
RGB1_PATH = "ganspace/notebooks/data/steerability/stylegan_ffhq/ffhq_rgb_1.npy"
RGB2_PATH = "ganspace/notebooks/data/steerability/stylegan_ffhq/ffhq_rgb_2.npy"
#Biggan deep
LINEAR_SHIFT_PATH = "ganspace/notebooks/data/steerability/biggan_deep_512/gan_steer-linear_shiftx_512.pkl"
LINEAR_ZOOM_PATH = "ganspace/notebooks/data/steerability/biggan_deep_512/gan_steer-linear_zoom_512.pkl"
#Stylegan cars
ROTATE_PATH = "ganspace/notebooks/data/steerability/stylegan_cars/rotate2d.npy"
SHIFTY_PATH = "ganspace/notebooks/data/steerability/stylegan_cars/shifty.npy"


# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load StyleGAN3 model
def load_stylegan3():
    with open(MODEL_PATH, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()  # Load the generator
    return G


# Generate a random latent vector
def generate_latent(G, truncation=0.5):
    z = torch.randn(1, G.z_dim).cuda()  # Generate latent in Z-space
    w = G.mapping(z, None)  # Map Z to W-space

    # Apply truncation
    w_avg = G.mapping.w_avg
    w = w_avg + truncation * (w - w_avg)

    # Force correct W+ shape (1, 16, 512)
    w_plus = w[:, :16, :]  # Slice only first 16 layers if more exist

    #print("Generated latent shape:", w_plus.shape)  # Debugging
    return w_plus


# Generate an image
def generate_image(G, latent, output_path):
    img = G.synthesis(latent, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) / 2 * 255  # Normalize
    img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    from PIL import Image
    Image.fromarray(img).save(output_path)


# Modify latent with InterfaceGAN
def edit_with_interfacegan(latent, direction_path, factor=10.0):
    direction = np.load(direction_path)

    # Ensure direction shape is (512,) or (1, 512)
    if len(direction.shape) == 1:
        direction = direction[np.newaxis, :]  # Convert (512,) to (1, 512)

    direction = torch.tensor(direction).cuda()

    # Expand direction to (1, 16, 512) to match latent
    direction = direction.unsqueeze(1).repeat(1, 16, 1)  # Expand along 16 layers

    #print("Direction shape after expansion:", direction.shape)  # Debugging
    #print("Latent shape before edit:", latent.shape)  # Debugging

    new_latent = latent + factor * direction  # Apply edit

    #print("Latent shape after edit:", new_latent.shape)  # Debugging
    return new_latent


def edit_with_ganspace(latent, direction_path, component_index, strength=3.0):
    pca_components = np.load(direction_path)
    pca_components = torch.tensor(pca_components).cuda()
    direction = pca_components[component_index]
    new_latent = latent + strength * direction
    return new_latent

if __name__ == "__main__":
    G = load_stylegan3()
    latent = generate_latent(G)
    
    original_img_path = os.path.join(OUTPUT_DIR, "original.png")

    #Test if direction is (1, 512)
    direction = np.load(AGE_PATH)
    print("Direction shape:", direction.shape)

    #print("Final edited latent shape:", edited_latent.shape)
    generate_image(G, latent, original_img_path)
    print(f"Generated image saved at {original_img_path}")

    # Reset pose before making edits
    #reset_pose_latent = edit_with_interfacegan(latent, POSE2_PATH, factor=0.0)  # Neutral position

    # Now apply different poses
    pose_variations = [(-6, "pose_left", POSE_PATH), (6, "pose_right", POSE_PATH), (-6, "pose_tilt_down", POSE2_PATH), (6, "pose_tilt_up", POSE2_PATH), (4, "shifty", SHIFTY_PATH), (-4, "rotate", SHIFTY_PATH)]

    for factor, name, path in pose_variations:
        edited_latent = edit_with_interfacegan(latent, path, factor=factor)
        pose_img_path = os.path.join(OUTPUT_DIR, f"edited_{name}.png")
        generate_image(G, edited_latent, pose_img_path)
        print(f"Edited pose image saved at {pose_img_path}")


    # Lighting variations using GANSpace
    for i, (comp, strength) in enumerate([(4, 3), (5, 3), (6, 3)]):
        edited_latent = edit_with_ganspace(latent, RGB_PATH, component_index=comp, strength=strength)
        light_img_path = os.path.join(OUTPUT_DIR, f"edited_light_{i}.png")
        generate_image(G, edited_latent, light_img_path)
        print(f"Edited lighting image saved at {light_img_path}")

    # Expressions
    smile_latent = edit_with_interfacegan(latent, SMILE_PATH, factor=8.0)  
    smile_img_path = os.path.join(OUTPUT_DIR, "edited_smile.png")
    generate_image(G, smile_latent, smile_img_path)
    print(f"Edited smile image saved at {smile_img_path}")

    #Generate an image where the subject is older
    old_img_path = os.path.join(OUTPUT_DIR, "edited_old.png")
    edited_latent = edit_with_interfacegan(latent, AGE_PATH, factor=7.0)  # Change factor as you'd like
    generate_image(G, edited_latent, old_img_path)
    print(f"Edited image saved at {old_img_path}")

    #Generate an image where the subject is younger
    young_img_path = os.path.join(OUTPUT_DIR, "edited_young.png")
    edited_latent = edit_with_interfacegan(latent, AGE_PATH, factor=-7.0)  # Change factor as you'd like
    generate_image(G, edited_latent, young_img_path)
    print(f"Edited image saved at {young_img_path}")
