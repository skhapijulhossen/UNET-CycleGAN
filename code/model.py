import torch
import torch.nn as nn
import numpy as np

# For UNET
class DoubleConv(nn. Module):
    def __init__(self, in_channels, out_channels):
        super (DoubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d (out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d (out_channels),
            nn.ReLU(inplace=True),
        )


    def forward (self, x):
        """Residual CONV"""
        x = self.conv1(x)
        return  x + self.conv2(x)


class UNET(nn.Module):
    """
    UNET Module
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super (UNET, self).__init__()
        self.upsample_steps = nn.ModuleList()
        self.downsample_steps = nn.ModuleList()
        self.pool = nn. MaxPool2d (kernel_size=2, stride=2)

        # DownsamplingBlock of UNET
        for feature in features:
            self.downsample_steps.append (DoubleConv(in_channels, feature))
            in_channels = feature

        # ##
        self.bottleneck = DoubleConv(features[-1], features [-1]*2)

        # UpsamplingBlock of UNET
        for feature in reversed(features):
            self.upsample_steps.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))

            self.upsample_steps.append(DoubleConv(feature*2, feature))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # downsample
        for downsample in self.downsample_steps:
            x = downsample(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck - before upsample
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # upsample
        for index in range(0, len(self.upsample_steps), 2):
            x = self.upsample_steps[index](x)
            skip_connection = skip_connections[index//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.upsample_steps[index+1](concat_skip)
        return self.final_conv(x)




# ViT
def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches


def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result


class MSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])


class ViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(ViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class ViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=1):
        # Super constructor
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([ViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        # 5) Classification FCN
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d),
            nn.Sigmoid()
        )

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
        return out
        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(out)


# LSB Steganography
class StegoEmbeddingLayer(nn.Module):
    def __init__(self):
        super(StegoEmbeddingLayer, self).__init__()

    def forward(self, image, text):
        """
        Embeds text into the RGB image using LSB steganography.

        Parameters:
        - image (torch.Tensor): RGB image tensor of shape (batch_size, 3, height, width).
        - text (str): The text to be embedded into the image.

        Returns:
        - torch.Tensor: Embedded image tensor.
        """
        batch_size, channels, height, width = image.size()

        # Text => binary
        binary_text = ''.join(bin(ord(char)) for char in text)
        binary_text += '1111111111111110'  # delimiter

        binary_index = 0

        # Embed the text into LSB
        for i in range(batch_size):
            for j in range(channels):
                for k in range(height):
                    for l in range(width):
                        pixel_value = image[i, j, k, l]
                        binary_pixel_value = format(pixel_value.item(), '08b')

                        # Modify the LSB to encode the text
                        binary_pixel_value = binary_pixel_value[:-1] + binary_text[binary_index]
                        binary_index = (binary_index + 1) % len(binary_text)

                        # Convert the binary pixel back to decimal
                        embedded_pixel_value = int(binary_pixel_value[:-1], 2)

                        image[i, j, k, l] = embedded_pixel_value

        return image

# Tests
def test_Disc():
    x = torch.randn((5, 3, 256, 256))
    # model = Discriminator(in_channels=3)
    model = ViT((3, 256, 256), n_patches=16, n_blocks=2, hidden_d=8, n_heads=2, out_d=1)
    preds = model(x)
    print(preds.shape)


def test_Gen():
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = UNET(img_channels)
    print(gen(x).shape)


if __name__ == '__main__':
    test_Disc()
    test_Gen()
    print("Done")