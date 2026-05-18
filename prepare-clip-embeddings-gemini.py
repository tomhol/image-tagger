import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path

# Configuration
INPUT_ROOT = Path("training-set")
OUTPUT_FILE = "dog_embeddings_balanced.npy"
MODEL_NAME = "ViT-B/32"

def generate_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load CLIP model
    model, preprocess = clip.load(MODEL_NAME, device=device)
    
    dog_embeddings = {}
    
    # Process each subfolder (raff, saga, ...)
    for dog_name in ["Raff", "Saga", "Cakun"]:
        input_dir = INPUT_ROOT / dog_name
        if not input_dir.exists():
            print(f"Directory {input_dir} not found. Skipping.")
            continue
            
        image_paths = list(input_dir.glob("*.JPG")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        if not image_paths:
            print(f"No images found for {dog_name}.")
            continue
            
        print(f"Generating embeddings for {dog_name} ({len(image_paths)} images)...")
        
        all_embeddings = []
        
        with torch.no_grad():
            for img_path in image_paths:
                try:
                    # Load and preprocess image
                    image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                    # Encode image
                    image_features = model.encode_image(image)
                    # Normalize features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    all_embeddings.append(image_features.cpu().numpy())
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        if all_embeddings:
            # Average the embeddings to get a single representative vector for each dog
            avg_embedding = np.mean(np.vstack(all_embeddings), axis=0)
            # Re-normalize the average embedding
            avg_embedding /= np.linalg.norm(avg_embedding)
            dog_embeddings[dog_name] = avg_embedding
            print(f"Successfully generated reference embedding for {dog_name}.")

    # Save the dictionary of embeddings
    if dog_embeddings:
        np.save(OUTPUT_FILE, dog_embeddings)
        print(f"Embeddings saved to {OUTPUT_FILE}")
    else:
        print("No embeddings were generated.")

if __name__ == "__main__":
    generate_embeddings()
