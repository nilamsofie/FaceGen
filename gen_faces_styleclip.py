import os
import torch
import numpy as np
import pickle
from PIL import Image
import clip
import sys
import time
from tqdm import tqdm
import hashlib
import argparse


# Configuration
STYLEGAN3_DIR = "stylegan3"
MODEL_PATH = "stylegan3/models/stylegan3-r-ffhq-1024x1024.pkl"
OUTPUT_ROOT = "clip_outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Add StyleGAN3 to path
sys.path.append(STYLEGAN3_DIR)

def load_models():
    """Load both StyleGAN3 and CLIP models"""
    with open(MODEL_PATH, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()
    clip_model, _ = clip.load("ViT-B/32", device="cuda")
    return G, clip_model

def generate_random_latent(G, truncation=0.7):
    """Generate a new random face with different seed each time"""
    # Create new seed based on current time
    seed = int(time.time() * 1000) % 2**32
    torch.manual_seed(seed)
    
    z = torch.randn(1, G.z_dim).cuda()
    w = G.mapping(z, None, truncation_psi=truncation)
    return w[:, :16, :]  # Return W+ space (1, 16, 512)

def generate_image(G, latent, path):
    """Generate image from latent"""
    with torch.no_grad():
        img = G.synthesis(latent, noise_mode='const')
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    Image.fromarray(img[0].cpu().numpy(), 'RGB').save(path)
    return img

def styleclip_edit(
    G,
    clip_model,
    original_latent,
    text_prompt,
    steps=150,
    lr=0.02,
    clip_lambda=1.0,
    identity_lambda=1.2,
    l2_lambda=0.2,
    style_lambda=0.5,
    truncation=0.7
):
    """StyleCLIP editing with strong identity preservation"""
    device = "cuda"
    
    # Make latent trainable
    latent = original_latent.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=lr)
    
    # Encode text prompt
    text_tokens = clip.tokenize([text_prompt]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens).detach()
    
    # Store original image features
    with torch.no_grad():
        original_img = G.synthesis(original_latent, noise_mode='const')
        original_img = (original_img + 1) / 2  # Normalize to [0,1]
        original_resized = torch.nn.functional.interpolate(original_img, size=224, mode='bilinear')
        original_features = clip_model.encode_image(original_resized).detach()
    
    # Optimization loop
    for step in tqdm(range(steps), desc=f"Editing '{text_prompt[:20]}...'"):
        optimizer.zero_grad()
        
        # Generate current image
        current_img = G.synthesis(latent, noise_mode='const')
        current_img = (current_img + 1) / 2  # Normalize to [0,1]
        
        # Prepare for CLIP
        img_resized = torch.nn.functional.interpolate(current_img, size=224, mode='bilinear')
        img_features = clip_model.encode_image(img_resized)
        
        # Calculate multiple loss components
        clip_loss = 1 - torch.cosine_similarity(text_features, img_features)
        identity_loss = 1 - torch.cosine_similarity(original_features, img_features)
        l2_loss = torch.norm(latent - original_latent, 2)
        style_loss = torch.norm(img_features - original_features, 2)
        
        # Combined loss (weighted sum)
        total_loss = (clip_lambda * clip_loss +
                     identity_lambda * identity_loss +
                     l2_lambda * l2_loss +
                     style_lambda * style_loss)
        
        total_loss.backward()
        optimizer.step()
        
        # Apply truncation to maintain realism
        if truncation < 1.0:
            with torch.no_grad():
                latent.data = G.mapping.w_avg + truncation * (latent.data - G.mapping.w_avg)
    
    return latent.detach()

def create_edits(G, clip_model, original_latent, output_dir):
    """Create all edits for the generated face"""
    edits = []
    os.makedirs(output_dir, exist_ok=True)
    
    # Save original image
    original_path = os.path.join(output_dir, "original.png")
    generate_image(G, original_latent, original_path)
    edits.append(("Original", original_path))
    
    # Define edits with carefully tuned parameters
    edit_definitions = [
        ("person smiling with teeth showing", "smile", 200, 0.1, 1, 2, 0.1, 1),
        ("a person in full right profile view with face turned away", "look_right", 300, 0.12, 1.6, 1.0, 0.02, 0.85),
        ("a person in full left profile view with face turned away", "look_left", 300, 0.12, 1.6, 1.0, 0.02, 0.85),
        ("a young childlike person with smooth skin and bright eyes", "young", 200, 0.05, 0.8, 2, 0.1, 0.9),
        ("an old person with wrinkles, grey hair, and sagging skin", "old", 250, 0.07, 1, 1.5, 0.05, 0.85),
        ("a person under very soft, bright studio lighting with no shadows", "soft_light", 300, 0.12, 1.5, 1.0, 0.02, 0.85),
        ("a person under extreme dramatic lighting with harsh shadows and bright highlights", "dramatic_light", 350, 0.15, 1.8, 0.8, 0.02, 0.8),
        ("a person with intense backlighting, glowing light behind the head casting shadows on the face", "backlight", 350, 0.15, 1.8, 0.8, 0.02, 0.8)
    ]
    
    for prompt, name, steps, lr, clip_l, id_l, l2_l, style_l in edit_definitions:
        try:
            edited_latent = styleclip_edit(
                G, clip_model, original_latent,
                text_prompt=prompt,
                steps=steps,
                lr=lr,
                clip_lambda=clip_l,
                identity_lambda=id_l,
                l2_lambda=l2_l,
                style_lambda=style_l
            )
            edit_path = os.path.join(output_dir, f"{name}.png")
            generate_image(G, edited_latent, edit_path)
            edits.append((prompt, edit_path))
        except Exception as e:
            print(f"Failed to create {name} edit: {str(e)}")
    
    return edits, output_dir


def hash_latent(latent):
    """Generate a short hash of the latent tensor"""
    latent_np = latent.detach().cpu().numpy()
    latent_bytes = latent_np.tobytes()
    return hashlib.sha256(latent_bytes).hexdigest()[:10]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_subjects', type=int, default=1, help="Number of subjects to generate")
    return parser.parse_args()


def main():
    args = parse_args()
    G, clip_model = load_models()

    for i in range(args.n_subjects):
        print(f"\n🔄 Generating subject {i + 1} of {args.n_subjects}...")
        original_latent = generate_random_latent(G)
        subject_hash = hash_latent(original_latent)
        subject_dir = os.path.join(OUTPUT_ROOT, f"subject_{subject_hash}")

        edits, output_folder = create_edits(G, clip_model, original_latent, subject_dir)

        print(f"\n✅ Subject saved in: {output_folder}")
        for desc, path in edits:
            print(f"- {desc}: {path}")


if __name__ == "__main__":
    main()