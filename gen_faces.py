import sys
import os
import numpy as np
import torch
import pickle
import argparse
import hashlib
from pathlib import Path
from PIL import Image

# Directories and paths
STYLEGAN3_DIR = "stylegan3"
MODEL_PATH = "stylegan3/models/stylegan3-r-ffhq-1024x1024.pkl"
OUTPUT_DIR = "outputs"

# InterFaceGAN Boundaries
AGE_PATH = "boundaries/stylegan_ffhq_age_w_boundary.npy"
POSE_PATH = "boundaries/stylegan_ffhq_pose_w_boundary.npy"
POSE2_PATH = "boundaries/stylegan_ffhq_pose_boundary.npy"
SMILE_PATH = "boundaries/stylegan_ffhq_smile_w_boundary.npy"

# GANSpace directions
RGB_PATH = "steerability/stylegan_ffhq/ffhq_rgb_0.npy"

# Ensure paths work
sys.path.append(STYLEGAN3_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load StyleGAN3 model
def load_stylegan3():
    print("Loading StyleGAN3 model...")
    with open(MODEL_PATH, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    print("Model loaded.")
    return G

# Generate a random latent vector
def generate_latent(G, truncation=0.5):
    z = torch.randn(1, G.z_dim).cuda()
    w = G.mapping(z, None)
    w_avg = G.mapping.w_avg
    w = w_avg + truncation * (w - w_avg)
    w_plus = w[:, :16, :]
    return w_plus

# Generate an image
def generate_image(G, latent, output_path):
    img = G.synthesis(latent, noise_mode='const')
    img = (img.clamp(-1, 1) + 1) / 2 * 255
    img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)[0]
    Image.fromarray(img).save(output_path)

# Edit latent with InterfaceGAN
def edit_with_interfacegan(latent, direction_path, factor=10.0):
    direction = np.load(direction_path)
    if len(direction.shape) == 1:
        direction = direction[np.newaxis, :]
    direction = torch.tensor(direction).cuda()
    direction = direction.unsqueeze(1).repeat(1, 16, 1)
    return latent + factor * direction

# Modify latent with GanSpace
def edit_with_ganspace(latent, direction_path, component_index, strength=3.0):
    pca_components = np.load(direction_path)
    pca_components = torch.tensor(pca_components).cuda()
    direction = pca_components[component_index]
    return latent + strength * direction

# Hash latent for folder naming
def latent_to_hash(latent_tensor):
    latent_np = latent_tensor.detach().cpu().float().numpy()
    latent_bytes = latent_np.tobytes()
    latent_hash = hashlib.sha256(latent_bytes).hexdigest()
    return latent_hash[:10]

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_subjects', type=int, default=1)
    args = parser.parse_args()

    G = load_stylegan3()

    for i in range(args.n_subjects):
        try:
            latent = generate_latent(G)
            latent_hash = latent_to_hash(latent)
            subject_dir = os.path.join(OUTPUT_DIR, f"subject_{latent_hash}")
            os.makedirs(subject_dir, exist_ok=True)
            print(f"[{i+1}/{args.n_subjects}] Generating subject: {subject_dir}")

            # Save latent
            np.save(os.path.join(subject_dir, "latent.npy"), latent.detach().cpu().numpy())

            # Original image
            generate_image(G, latent, os.path.join(subject_dir, "original.png"))

            # Poses
            pose_variations = [
                (-6, "pose_left", POSE_PATH),
                (6, "pose_right", POSE_PATH),
                (-6, "pose_tilt_down", POSE2_PATH),
                (6, "pose_tilt_up", POSE2_PATH)
            ]
            for factor, name, path in pose_variations:
                edited_latent = edit_with_interfacegan(latent, path, factor)
                generate_image(G, edited_latent, os.path.join(subject_dir, f"edited_{name}.png"))

            # Lighting
            for j, (comp, strength) in enumerate([(4, 3), (5, 3), (6, 3)]):
                edited_latent = edit_with_ganspace(latent, RGB_PATH, comp, strength)
                generate_image(G, edited_latent, os.path.join(subject_dir, f"edited_light_{j}.png"))

            # Smile
            smile_latent = edit_with_interfacegan(latent, SMILE_PATH, 8.0)
            generate_image(G, smile_latent, os.path.join(subject_dir, "edited_smile.png"))

            # Age
            old_latent = edit_with_interfacegan(latent, AGE_PATH, 7.0)
            young_latent = edit_with_interfacegan(latent, AGE_PATH, -7.0)
            generate_image(G, old_latent, os.path.join(subject_dir, "edited_old.png"))
            generate_image(G, young_latent, os.path.join(subject_dir, "edited_young.png"))

        except Exception as e:
            print(f"Error generating subject {i}: {e}")
