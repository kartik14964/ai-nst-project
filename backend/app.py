import os
import sys
import torch
import gradio as gr
from PIL import Image
from torchvision import transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# ZeroGPU decorator for free Hugging Face GPU acceleration
try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

from utils.models import VGGEncoder, Decoder
from utils.utils import adaptive_instance_normalization

vgg_path = os.path.join(BASE_DIR, 'vgg_normalised.pth')
decoder_path = os.path.join(BASE_DIR, 'experiment', 'final_exp', 'decoder_final.pth')

encoder = VGGEncoder(vgg_path)
encoder.eval()

decoder = Decoder()
decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))
decoder.eval()

@spaces.GPU
def style_transfer(content_img, style_img, alpha=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = encoder.to(device)
    dec = decoder.to(device)

    tf = transforms.Compose([transforms.Resize(512), transforms.ToTensor()])

    if not isinstance(content_img, Image.Image):
        content_img = Image.fromarray(content_img)
    if not isinstance(style_img, Image.Image):
        style_img = Image.fromarray(style_img)

    content_tensor = tf(content_img.convert('RGB')).unsqueeze(0).to(device)
    style_tensor = tf(style_img.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        content_feats = enc(content_tensor, is_test=True)
        style_feats = enc(style_tensor, is_test=True)
        stylized_feats = adaptive_instance_normalization(content_feats, style_feats)
        stylized_feats = float(alpha) * stylized_feats + (1.0 - float(alpha)) * content_feats
        output = dec(stylized_feats)

    return transforms.ToPILImage()(output.cpu().squeeze(0).clamp(0, 1))

demo = gr.Interface(
    fn=style_transfer,
    inputs=[
        gr.Image(label="Content Image", type="pil"),
        gr.Image(label="Style Image", type="pil"),
        gr.Slider(minimum=0.0, maximum=1.0, value=1.0, step=0.05, label="Alpha")
    ],
    outputs=gr.Image(label="Stylized Output", type="pil"),
    title="AdaIN Neural Style Transfer API"
)

if __name__ == '__main__':
    demo.launch()

