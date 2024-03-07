import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

# Define the LSB embedding layer
class LSBEmitter(nn.Module):
    def __init__(self, message_size):
        super(LSBEmitter, self).__init__()
        self.message_size = message_size

    def forward(self, cover_image, secret_data):
        # Flatten the cover image
        cover_flat = cover_image.reshape(-1)
        
        # Convert the secret data to binary
        secret_binary = torch.cat([torch.tensor([int(bit) for bit in format(byte, '08b')]) for byte in secret_data])

        # Ensure the cover image has enough capacity for the secret data
        if len(secret_binary) > len(cover_flat):
            raise ValueError("Insufficient capacity in the cover image for the secret data.")

        # Embed the secret data into the LSBs of the cover image
        cover_flat[:len(secret_binary)] = (cover_flat[:len(secret_binary)] & 0b11111110) | secret_binary

        # Reshape the cover image to its original shape
        stego_image = cover_flat.reshape(cover_image.shape)
        
        return stego_image



if __name__ == "__main__":
    cover_image_path = r'D:\CycleGAN\data\test\horses\n02381460_40.jpg'
    cover_image = Image.open(cover_image_path)
    cover_image_tensor = torch.tensor(np.array(cover_image), dtype=torch.uint8).permute(2, 0, 1).float() / 255.0

    # Convert a secret message to bytes
    secret_message = "secret message"
    secret_data = bytearray(secret_message, 'utf-8')

    # Initialize the LSBEmitter with the message size
    lsb_emitter = LSBEmitter(message_size=len(secret_data) * 8)

    # Embed the secret data into the cover image
    stego_image = lsb_emitter(cover_image_tensor, torch.ByteTensor(secret_data))

    # Display the cover and stego images
    Image.fromarray((cover_image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).show(title='Cover Image')
    Image.fromarray((stego_image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)).show(title='Stego Image')
