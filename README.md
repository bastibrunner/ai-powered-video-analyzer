

### **📝 Understanding the Power of This Project**
Imagine being able to **decode the essence of a video**—not just frame by frame, but capturing its **story, emotions, and context**—without ever connecting to the internet. 

This project does exactly that.

It brings together the **best AI models** in **object detection, image captioning, speech recognition, and audio event detection**—all running **locally** on your personal computer.

With this tool, a **documentary becomes a narrated masterpiece**, a **silent clip finds its meaning**, and a **conversation in a foreign language becomes readable text**. All of this happens **offline, using AI directly on your machine**.

Let’s break down why this matters:

- **Ever watched a video and wished you could instantly get a summary?** 
- **Ever needed to process a large collection of footage but had no time to sit through everything?** 
- **Ever wanted a tool that works **without cloud servers**, ensuring **total privacy**?**  

With **this project**, a **fighter jet documentary**, a **comedic skit**, or even an **experiment in extreme weather conditions** can be summarized **intelligently**, capturing the details that matter.

It doesn’t just **describe what’s happening**; it **understands the video’s story**.

---

## 📌 **Key Features**
✅ **Fully Offline** – No internet required after initial model downloads.  
✅ **Object Detection** – Identifies objects with **YOLO**.  
✅ **Scene Description** – Generates image captions with **BLIP**.  
✅ **Speech Transcription** – Converts spoken words into text with **Whisper**.  
✅ **Audio Event Detection** – Recognizes **sounds, music, and environmental noises** using **PANNs**.  
✅ **AI Summarization** – Uses **powerful LLMs (via Ollama)** to create **meaningful, human-like video summaries**.  
✅ **Graphical User Interface (GUI)** – Simple and intuitive **Tkinter-based** interface.  
✅ **Supports Multiple Languages** – Works in **English, Persian, and more**.  

---

## 🚀 **Installation Guide**
### 1️⃣ Install Required Packages

#### Using `uv` (recommended):
```bash
uv sync
```

#### Using pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2️⃣ Install **Ollama** (For AI Summarization)

#### Linux / macOS:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

#### Windows (PowerShell):
```powershell
iwr -useb https://ollama.com/install.ps1 | iex
```

---

### 3️⃣ Download Required AI Models

#### 🔹 Ollama LLMs:
```bash
ollama pull phi4:latest
ollama pull partai/dorna-llama3:latest
ollama pull qwen:14b
```

#### 🔹 YOLO (Object Detection)
- Download from **[YOLOv8 weights](https://github.com/ultralytics/ultralytics)**  
- Save in the **project directory**.

#### 🔹 Whisper (Speech-to-Text)
```python
import whisper
whisper.load_model("large-v2")
```

#### 🔹 BLIP (Image Captioning)
Automatically downloaded when used.

#### 🔹 PANNs (Audio Analysis)
- Download **`cnn14.pth`** from:  
  **[PANNs GitHub Repository](https://github.com/qiuqiangkong/audioset_tagging_cnn)**  
- Place it in:  
  ```plaintext
  models/cnn14.pth
  ```

---

### 4️⃣ **Windows Users Only** – Setup **Tesseract OCR**
- Install **Tesseract OCR** from:  
  [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Set the path in `pytesseract.pytesseract.tesseract_cmd`.

### 5️⃣ System dependencies (macOS / Linux)
- **ffmpeg**: required by MoviePy for video/audio IO.
- **tesseract**: required by `pytesseract` (binary, not a Python package).

On macOS with Homebrew:

```bash
brew install ffmpeg tesseract
```

If `tesseract` isn’t on your `PATH`, you can set `TESSERACT_CMD` (or just ensure `brew` puts it on `PATH`).

---

## 🎬 **How to Use the Video Analyzer**
### **Option 1: Run via GUI (Recommended)**
```bash
python video_processing_gui.py
```
- 📁 **Load a video**
- 🗣️ **Choose transcription language**
- 🔮 **Pick an AI model for summarization**
- ▶ **Click "Start Processing"**

### **Option 2: Run via CLI (Command Line Mode)**
```bash
python video_processing.py --video path/to/video.mp4 --save
```



---

## 🖥 **System Requirements**
🔹 **Recommended GPU:** NVIDIA RTX 3060 / 3070 or higher  
🔹 **RAM:** Minimum **16GB**, recommended **32GB**  
🔹 **Disk Space:** At least **20GB** free for models  

---

## 🔍 **How This Works (Step-by-Step)**
1️⃣ **Extracts audio** from the video.  
2️⃣ **Transcribes speech** using Whisper.  
3️⃣ **Detects objects** using YOLO.  
4️⃣ **Generates captions** using BLIP.  
5️⃣ **Identifies sound events** using PANNs.  
6️⃣ **Summarizes the video** using LLM (via Ollama).  
7️⃣ **Creates a final annotated video and text report**.  

---

## 💡 **Why This Matters**
### **Not Just a Description – A True AI Narrative**
This tool **doesn’t just list objects** in a video—it **understands the context** and **summarizes the story behind it**.

For example, instead of saying:

> *"A plane is in the sky."*

It could summarize:

> *"The F-35 Lightning II is seen performing aerial maneuvers, showcasing its speed, stealth, and cutting-edge avionics."*

Instead of just describing actions:

> *"A teacher and student are talking."*

It could recognize humor and say:

> *"A comedic skit unfolds as a student playfully challenges his teacher, using modern technology as a witty response to traditional education."*

Instead of **adding fictional details** to an experiment:

> *"A person is holding a glass."*

It would **describe the scene realistically** and say:

> *"A person stands in a snowy landscape, holding a glass of hot water. As they throw the water into the freezing air, fine mist and ice crystals form instantly, demonstrating the Mpemba effect in extreme cold."*

However, **AI isn’t perfect**—one model mistakenly **detected a dog in the scene**, even though there wasn’t one! This highlights how **AI can sometimes misinterpret visuals**, but it is constantly improving in accuracy.

This means **richer, more meaningful insights**—whether you're analyzing a **documentary**, a **funny video**, or a **scientific experiment**—while also showing the challenges of AI **understanding complex scenes perfectly**.

---

## 🤝 **Contributing**
This project is **open-source**.  
Want to improve it? **Fork the repo, contribute, or suggest features!**  

---

## 📜 **License**
Licensed under the **MIT License**.

---

## 🙌 **Acknowledgments**
Thanks to:
- **Ultralytics** (YOLO)
- **OpenAI** (Whisper)
- **Salesforce** (BLIP)
- **Hugging Face** (Model Hosting)
- **Ollama** (On-device LLMs)
- **Dr. Mark Eramian** and the **Image Lab in the Department of Computer Science at the University of Saskatchewan**, where I have had the opportunity to deepen my knowledge in computer vision.  
  His mentorship has not only shaped my technical understanding of the field but has also guided me in approaching research with integrity, critical thinking, and a strong ethical foundation.



---

### **✨ Final Thoughts**
This is **not just a video processing tool**—it’s a **local AI-powered storytelling engine**.

🚀 **Turn your raw videos into AI-generated narratives.**  
🔒 **Keep your data private.**  
🧠 **Understand your videos like never before.**  

---

## 👤 **Credits**
This project was developed by **Arash Sajjadi** as part of a **home research initiative** to explore the capabilities of AI in **video understanding, transcription, and summarization**—all while keeping everything **offline** and private.

📌 Connect with me on **[LinkedIn](https://www.linkedin.com/in/arash-sajjadi/)**.

-----

## 📖 References
For more details on the AI models used in this project, see:

- **YOLOv11 (Object Detection)**: A state-of-the-art real-time detection model with improved feature extraction and efficiency.  
  *Khanam, R., & Hussain, M. (2024).* [arXiv:2410.17725](https://arxiv.org/abs/2410.17725)

- **Whisper (Speech Recognition)**: A robust model for multilingual speech-to-text conversion.  
  *Radford, A., et al. (2022).* [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)

- **BLIP (Image Captioning)**: A vision-language model that generates natural language descriptions of images.  
  *Li, J., et al. (2022).* [arXiv:2201.12086](https://arxiv.org/abs/2201.12086)

- **PANNs (Audio Event Detection)**: A neural network designed to recognize environmental sounds and music.  
  *Kong, Q., et al. (2020).* IEEE/ACM Transactions on Audio, Speech, and Language Processing.  




