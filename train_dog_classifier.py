import os
# Configure Keras to use PyTorch backend as requested
os.environ["KERAS_BACKEND"] = "torch"

import keras
import torch
# Ensure Keras uses channels_last (H, W, C) to match model definition and inference script
# even though PyTorch backend usually defaults to channels_first.
keras.config.set_image_data_format("channels_last")
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from keras import layers
from pathlib import Path

# Configuration
DATA_DIR = Path("training-set")
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 10
MODEL_SAVE_PATH = "dog_classifier.keras"
CLASSES_SAVE_PATH = "dog_classes.txt"

def main():
    if not DATA_DIR.exists():
        print(f"Error: Directory '{DATA_DIR}' not found. Please ensure your 'training-set' folder exists.")
        return

    print(f"Loading data from {DATA_DIR}...")
    
    # Use torchvision for data loading to avoid TensorFlow dependency
    # EfficientNetV2 expects inputs in [0, 255] range. 
    # ToTensor() converts to [0, 1], so we scale back to [0, 255].
    data_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.permute(1, 2, 0)), # Convert (C, H, W) -> (H, W, C)
        transforms.Lambda(lambda x: x * 255.0)
    ])

    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=data_transform)
    
    # Split 80/20
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    
    # Use fixed generator for reproducibility (equivalent to seed=123)
    gen = torch.Generator().manual_seed(123)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=gen)

    # Create DataLoaders (Keras 3 supports PyTorch DataLoaders directly)
    train_ds = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_ds = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = full_dataset.classes
    print(f"Classes found: {class_names}")

    # Save class names for the inference script
    with open(CLASSES_SAVE_PATH, "w") as f:
        for name in class_names:
            f.write(f"{name}\n")

    # Transfer Learning Strategy:
    # Use EfficientNetV2B0 - lightweight and includes internal preprocessing (rescaling)
    base_model = keras.applications.EfficientNetV2B0(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
        pooling="avg"
    )
    base_model.trainable = False  # Freeze base model

    # Build model
    inputs = keras.Input(shape=IMG_SIZE + (3,))
    x = base_model(inputs)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Starting training (optimized for CPU)...")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    print("Training complete.")

if __name__ == "__main__":
    main()