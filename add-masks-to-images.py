from PIL import Image, ImageChops
import os

# Paths to your folders
image_dir = '/mnt/c/Users/tomhol/Documents/Zephyr/brouk'
mask_dir = '/mnt/c/Users/tomhol/Documents/Zephyr/brouk'
output_dir = '/mnt/c/Users/tomhol/Documents/Zephyr/brouk2'

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        img = Image.open(os.path.join(image_dir, filename)).convert("RGB")
        # Assuming mask has the same name but .png extension
        mask = Image.open(os.path.join(mask_dir, filename.replace(".jpg", ".mask.png"))).convert("RGB")
        # If masks are smaller, resize to match the image size
        mask = mask.resize(img.size)
        
        # multiply the image with the mask so the masked areas become black
        img = ImageChops.multiply(img, mask)

        # and now save the file as jpg with the masked areas as solid black
        img = img.convert("RGB")

        # set the jpg quality to 95
        img.save(os.path.join(output_dir, filename), quality=95)
        print(f"Processed {filename}")