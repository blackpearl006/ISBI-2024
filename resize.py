from PIL import Image
import os

resize_to = (224, 224)

networks = ['default_mode', 'cerebellum', 'frontoparietal', 'occipital', 'cingulo-opercular','sensorimotor']
for network in networks:
    for folder in ['healthy', 'mci']:
        data_dir = f'/Users/ninad/Documents/_CBR/Data/RP/{network}/{folder}'
        output_dir = f'/Users/ninad/Documents/_CBR/Data/resized/{network}/{folder}'

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(data_dir):
            if filename.endswith(".png"):
                image = Image.open(os.path.join(data_dir, filename))
                image = image.resize(resize_to, Image.LANCZOS)
                image.save(os.path.join(output_dir, filename))

print("Image resizing complete.")