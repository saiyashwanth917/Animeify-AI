import io
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from torchvision.transforms.functional import to_tensor, to_pil_image

from model import Generator

st.set_page_config(page_title="Animeify AI", page_icon="🎨", layout="wide")

BASE = Path(__file__).parent
STYLES = {
    "Paprika (painted, colourful)": "paprika.pt",
    "Face Paint v2 (soft illustration)": "face_paint_512_v2.pt",
}


@st.cache_resource(show_spinner="Loading GAN model...")
def load_generator(weight_file: str):
    net = Generator()
    net.load_state_dict(torch.load(BASE / "weights" / weight_file, map_location="cpu"))
    return net.eval()


def stylize(img: Image.Image, weight_file: str) -> Image.Image:
    net = load_generator(weight_file)
    # network needs sizes divisible by 32
    w, h = img.size
    img = img.resize((max(32, w // 32 * 32), max(32, h // 32 * 32)), Image.LANCZOS)
    x = to_tensor(img).unsqueeze(0) * 2 - 1
    with torch.no_grad():
        y = net(x)[0].clamp(-1, 1) * 0.5 + 0.5
    return to_pil_image(y)


st.title("🎨 Animeify AI")
st.caption("Turn a room photo into an artistic version using a pretrained GAN (AnimeGANv2).")

style = st.sidebar.selectbox("Art style", list(STYLES))
uploaded = st.file_uploader("Upload a room photo", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded.read()))).convert("RGB")
    img.thumbnail((640, 640))
    left, right = st.columns(2)
    left.subheader("Original")
    left.image(img, width="stretch")
    with st.spinner("Generating..."):
        out = stylize(img, STYLES[style])
    right.subheader(" Animeify AI output")
    right.image(out, width="stretch")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    right.download_button("Download", buf.getvalue(), "Animeify AI.png", "image/png")
else:
    st.info("Upload a room photo to begin.")

with st.expander("About this project"):
    st.write(
        "Uses the pretrained AnimeGANv2 generator (MIT licence, by Bryan Lee). "
        "A GAN has a generator and a discriminator; at inference only the trained "
        "generator is needed, so it runs fast on a CPU."
    )
