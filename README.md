---
title: Virtual Goggles Try-On 👓
emoji: 🕶️
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.33.0"
app_file: app.py
pinned: false
---
🕶️ Virtual Goggles Try-On using AI & ML
👓 Try on goggles virtually — in real time or using an uploaded image — powered by MediaPipe FaceMesh and BiSeNet segmentation.
🎯 Project Overview

This project allows users to try goggles virtually using either their webcam or an uploaded photo.

It uses MediaPipe FaceMesh to detect 468 facial landmarks and BiSeNet (trained on CelebAMask-HQ) for face segmentation to handle occlusion — making the goggles appear naturally behind the ears, nose, or hair.

🧩 Features

🎥 Real-time face detection & landmark tracking

🖼️ Upload your own goggles (PNG with transparency)

⚙️ Automatic alignment — adjusts for face size, tilt, and rotation

✂️ Occlusion handling (hair, ears, and nose visibility)

🧠 Uses AI segmentation (BiSeNet) with fallback to landmark occlusion

💻 Streamlit-based web app — easy to run and deploy

🧩 Lightweight — runs even on CPU

🧠 Tech Stack
Component	Tool
Language	Python 3.9+
Framework	Streamlit
Landmark Detection	MediaPipe FaceMesh
Segmentation	BiSeNet (CelebAMask-HQ)
Computer Vision	OpenCV
Image Handling	Pillow (PIL)
Machine Learning	PyTorch
Utilities	NumPy
⚙️ Installation & Setup Instructions
1️⃣ Create a Python Environment
conda create -n goggles python=3.9 -y
conda activate goggles

2️⃣ Install Dependencies
pip install streamlit opencv-python mediapipe torch torchvision torchaudio pillow numpy


or (recommended)

pip install -r requirements.txt


(Make sure requirements.txt includes all libraries above.)

3️⃣ Download BiSeNet Weights

The segmentation model helps handle occlusion (hair, nose, ears covering goggles).

Visit: https://github.com/zllrunning/face-parsing.PyTorch

Scroll to "Pre-trained Models" section.

Download the file named 79999_iter.pth from the Google Drive link.

Rename it to bisenet.pth

Place it inside your project folder under:

models/bisenet.pth

4️⃣ Run the Application
streamlit run app.py


Then open your browser and go to:

👉 http://localhost:8501

🚀 Usage

Choose “Webcam (Real-time)” or “Upload Image (single)”.

Upload your goggles PNG (with transparent background).

Adjust:

Scale (size of goggles)

Vertical offset (fine-tuning)

Save snapshots directly from the app.

🧩 Folder Structure
VirtualTryOn/
│
├── app.py
├── models/
│   ├── bisenet_model.py
│   └── bisenet.pth
├── assets/
│   └── goggles1.png
├── requirements.txt
└── README.md

💡 Real-World Application

Solving a real e-commerce challenge:

🛒 Helps online shoppers try eyewear virtually before purchasing

🧍‍♂️ Reduces return rates and increases confidence in style choices

💻 Enables developers to build AR-based fashion and beauty apps

🎨 Can be extended to makeup, hairstyle, or mask try-ons

🧑‍💻 Team & Acknowledgements

Developed by:
👨‍💻 Srivass Kumar
🤖 ChatGPT (OpenAI) – Debugging, optimization, and architecture assistance

Inspired by:

Google’s MediaPipe FaceMesh

BiSeNet from “Bilateral Segmentation Network for Real-time Semantic Segmentation”

CelebAMask-HQ dataset

⚠️ Common Issues & Solutions
Issue	Cause	Fix
Goggles not appearing	Missing landmarks or transparent PNG issue	Ensure face is visible and goggles PNG has transparency
Segmentation weights not loading	Missing or wrong file name/path	Ensure models/bisenet.pth exists
Webcam not working	Permissions or driver issue	Allow camera access in browser
“Missing key(s)” in BiSeNet	State dict mismatch	Model still works — uses fallback occlusion
🌟 Future Enhancements

Support for multiple goggles styles dynamically

Add virtual glasses and masks

Enable mobile AR view (camera-based tracking)

Integrate 3D rotation tracking for realism

Export video try-on clips

❤️ Final Note

After 24 hours, 500+ debugs, and countless iterations —
this AI project finally works flawlessly.
It’s a showcase of persistence, learning, and teamwork.

Try it yourself 👇
Live Streamlit App
 (once deployed)

📸 Example Output

✅ Ready for Deployment
✅ Class Presentation-Ready
✅ AI-Powered AR Experience

