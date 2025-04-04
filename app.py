import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from torchvision import models
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
import av
from PIL import Image

# ------------------- Setup -------------------
st.set_page_config(page_title="CelebA ResNet18", layout="centered")
st.title("CelebA Attribute Detection with Two ResNet18 Models")

# Attribute names
celeba_attrs = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair", "Wearing_Earrings",
    "Wearing_Hat", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]

# ------------------- Session State Setup -------------------
if "top_k" not in st.session_state:
    st.session_state.top_k = 3

st.session_state.top_k = st.selectbox(
    "Top attributes to show", [1, 3, 5, 10],
    index=[1, 3, 5, 10].index(st.session_state.top_k)
)
top_k = st.session_state.top_k

# ------------------- Load Models -------------------
@st.cache_resource
def load_models():
    m1 = models.resnet18(pretrained=False)
    m1.fc = torch.nn.Linear(m1.fc.in_features, 40)
    m1.load_state_dict(torch.load("best_model_soft.pth", map_location='cpu'))
    m1.eval()

    m2 = models.resnet18(pretrained=False)
    m2.fc = torch.nn.Linear(m2.fc.in_features, 40)
    m2.load_state_dict(torch.load("best_model_attention.pth", map_location='cpu'))
    m2.eval()

    return m1, m2

model1, model2 = load_models()

# ------------------- Image Transform -------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ------------------- Inference Function -------------------
def predict_attributes(image):
    image = np.array(image.convert("RGB"))
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        out1 = torch.sigmoid(model1(input_tensor))[0]
        out2 = torch.sigmoid(model2(input_tensor))[0]

    top1 = torch.topk(out1, top_k)
    top2 = torch.topk(out2, top_k)

    pred1 = [(celeba_attrs[i], float(out1[i])) for i in top1.indices]
    pred2 = [(celeba_attrs[i], float(out2[i])) for i in top2.indices]

    return pred1, pred2, image

# ------------------- Mode Selection -------------------
mode = st.radio("Choose input method:", ["Upload Image", "Use Webcam"])

# ------------------- Webcam Option -------------------
if mode == "Use Webcam":
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.pred1 = []
            self.pred2 = []

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            input_tensor = transform(img_rgb).unsqueeze(0)

            with torch.no_grad():
                out1 = torch.sigmoid(model1(input_tensor))[0]
                out2 = torch.sigmoid(model2(input_tensor))[0]

            top1 = torch.topk(out1, top_k)
            top2 = torch.topk(out2, top_k)

            self.pred1 = [(celeba_attrs[i], float(out1[i])) for i in top1.indices]
            self.pred2 = [(celeba_attrs[i], float(out2[i])) for i in top2.indices]

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx = webrtc_streamer(
        key="celeba-cam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    st_autorefresh(interval=2000, key="refresh_sidebar")

    if ctx and ctx.state.playing and ctx.video_processor:
        st.sidebar.subheader("Model 1 Predictions")
        for attr, conf in ctx.video_processor.pred1:
            st.sidebar.write(f"{attr}: {conf:.2f}")

        st.sidebar.subheader("Model 2 Predictions")
        for attr, conf in ctx.video_processor.pred2:
            st.sidebar.write(f"{attr}: {conf:.2f}")

    else:
        st.sidebar.write("Waiting for webcam...")

# ------------------- Upload Option -------------------
else:
    uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        pred1, pred2, _ = predict_attributes(image)

        st.subheader("Model 1 Predictions")
        for attr, conf in pred1:
            st.write(f"{attr}: {conf:.2f}")

        st.subheader("Model 2 Predictions")
        for attr, conf in pred2:
            st.write(f"{attr}: {conf:.2f}")
