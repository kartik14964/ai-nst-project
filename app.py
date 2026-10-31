import os
import sys
from PIL import Image
import torch
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
nst_code_dir = os.path.join(BASE_DIR, 'NST_Code')
if nst_code_dir not in sys.path:
    sys.path.insert(0, nst_code_dir)

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

# ZeroGPU support on Hugging Face Spaces (falls back seamlessly if not available)
try:
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

# Load model weights on CPU initially
vgg_path = os.path.join(nst_code_dir, 'vgg_normalised.pth')
decoder_path = os.path.join(nst_code_dir, 'experiment', 'final_exp', 'decoder_final.pth')

encoder = VGGEncoder(vgg_path).to("cpu")
decoder = Decoder().to("cpu")
decoder.load_state_dict(torch.load(decoder_path, map_location="cpu"))

encoder.eval()
decoder.eval()

@gpu_decorator
def style_transfer(content_img, style_img, alpha=1.0):
    if content_img is None or style_img is None:
        return None

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(dev)
    decoder.to(dev)

    if not isinstance(content_img, Image.Image):
        content_img = Image.fromarray(content_img)
    if not isinstance(style_img, Image.Image):
        style_img = Image.fromarray(style_img)

    content_img = content_img.convert('RGB')
    style_img = style_img.convert('RGB')

    content_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])
    style_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor()
    ])

    content_tensor = content_transform(content_img).unsqueeze(0).to(dev)
    style_tensor = style_transform(style_img).unsqueeze(0).to(dev)
    style_tensor = style_transform(style_img).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = encoder(content_tensor, is_test=True)
        style_feats = encoder(style_tensor, is_test=True)

        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = alpha * stylized_feats + (1 - alpha) * content_feats

        stylized_tensor = decoder(stylized_feats)

    stylized_tensor = stylized_tensor.cpu().squeeze(0).clamp(0, 1)
    result_image = transforms.ToPILImage()(stylized_tensor)
    return result_image


import gradio as gr

# Example images
example_images = []
ex1 = [
    os.path.join(nst_code_dir, 'examples', 'brad_pitt.jpg'),
    os.path.join(nst_code_dir, 'examples', 'sketch.png'),
    1.0
]
ex2 = [
    os.path.join(nst_code_dir, 'examples', 'brad_pitt.jpg'),
    os.path.join(nst_code_dir, 'examples', 'picasso_seated_nude_hr.jpg'),
    1.0
]
if os.path.exists(ex1[0]) and os.path.exists(ex1[1]):
    example_images.append(ex1)
if os.path.exists(ex2[0]) and os.path.exists(ex2[1]):
    example_images.append(ex2)

demo = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(type="pil", label="Content Image (Photo)"),
        gr.Image(type="pil", label="Style Image (Artwork)"),
        gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=1.0,
            step=0.05,
            label="Style Strength (Alpha)",
            info="0.0 = content only, 1.0 = full stylization"
        )
    ],
    outputs=gr.Image(type="pil", label="Stylized Output"),
    title="🎨 Real-Time Neural Style Transfer with AdaIN",
    description="Upload a **Content Image** and a **Style Image**, adjust the style strength slider, and click **Submit** to generate artwork in real-time!",
    examples=example_images if example_images else None,
    theme=gr.themes.Soft(primary_hue="indigo")
)

if __name__ == '__main__':
    demo.launch()
