# 🔥 Jutsu Trainer

A real-time Naruto hand sign recognition system with visual jutsu effects, built with Python, MediaPipe, PyTorch, and Flask.

Perform Naruto hand signs in front of your webcam and watch jutsus come to life — Fireball, Chidori, Rasengan, Susanoo, and more — with particle effects, sound, and a Sharingan overlay.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-2.3+-black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-green)

---

## Features

- **Real-time hand sign detection** using a custom-trained CNN on top of MediaPipe hand landmarks
- **12 jutsus** with unique particle/visual effects rendered directly onto the webcam feed
- **Sharingan eye overlay** with Mangekyou awakening after 3 casts
- **Two-hand control** — one hand performs signs, the other controls effect size (pinch) and hue (wrist roll)
- **Practice mode** — free mode, perform any jutsu at any time
- **Training mode** — guided step-by-step hand sign trainer with sequence progress tracking
- **Generated sound effects** for every jutsu (or load your own WAV files)
- **Web UI** served via Flask, accessible from any device on your local network

---

## Jutsus

| Jutsu | English | Sequence |
|-------|---------|----------|
| Katon: Gokakyuu no Jutsu | Great Fireball | Horse → Tiger |
| Chidori | One Thousand Birds | Ox → Hare → Monkey → Bird → Dog |
| Rasengan | Spiralling Sphere | Ram → Snake → Horse |
| Kage Bunshin no Jutsu | Shadow Clone | Ram → Snake → Tiger |
| Sharingan | Copy Wheel Eye | Tiger → Boar → Ox → Dog |
| Katon: Ryuuka no Jutsu | Dragon Fire | Horse → Dragon → Tiger |
| Susanoo | — | Dragon → Bird → Ox *(Sharingan required)* |
| Suiton: Suiryuudan | Water Dragon Bullet | Dragon → Ram → Bird |
| Amaterasu | Goddess of the Sun | Dog → Tiger → Boar *(Sharingan required)* |
| Sabaku Kyu | Sand Shield | Dog → Boar → Ram |
| Kirin | Qilin | Sharingan + Chidori sequence |
| Mangekyou Sharingan | Kaleidoscope Eye | Cast 3 jutsus with Sharingan active |

---

## Requirements

- Python 3.10+
- Webcam
- CUDA-capable GPU *(optional but recommended for faster CNN inference)*

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/jutsu-trainer.git
cd jutsu-trainer
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install the CUDA version of PyTorch manually from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above, otherwise pip will install the CPU version.

### 4. Download MediaPipe model files

Download these two files and place them in the project root directory:

- [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
- [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)

### 5. Add the trained CNN model

Place your trained `jutsu_cnn.pth` file in the project root. The checkpoint must contain:

```python
{
  "model_state": ...,   # state dict for JutsuCNN
  "signs": [...],       # list of sign label strings
  "img_size": 64        # input image size (int)
}
```

### 6. (Optional) Add hand sign images for Training Mode

Create the folder `static/signs/` and add JPG images named after each sign:

```
static/signs/horse.jpg
static/signs/tiger.jpg
static/signs/ox.jpg
static/signs/hare.jpg
static/signs/monkey.jpg
static/signs/bird.jpg
static/signs/dog.jpg
static/signs/ram.jpg
static/signs/snake.jpg
static/signs/dragon.jpg
static/signs/boar.jpg
```

### 7. (Optional) Add custom sound effects

Place WAV files in the `sounds/` folder. If not present, sounds are generated procedurally at startup.

```
sounds/fireball.wav
sounds/chidori.wav
sounds/rasengan.wav
sounds/shadow_clone.wav
sounds/sharingan.wav
sounds/flamethrower.wav
sounds/susanoo.wav
sounds/kirin.wav
sounds/water_dragon.wav
sounds/amaterasu.wav
sounds/sand_shield.wav
sounds/mangekyou.wav
```

---

## Running

```bash
python main.py
```

Then open your browser and go to:

```
http://localhost:5000
```

To access from another device on the same WiFi network, use your machine's local IP instead:

```
http://192.168.x.x:5000
```

To find your local IP:
- **Windows:** `ipconfig` in Command Prompt
- **macOS/Linux:** `ifconfig` or `ip a` in Terminal

---

## Project Structure

```
jutsu-trainer/
├── main.py                  # Flask server + CV loop + all effects
├── requirements.txt
├── jutsu_cnn.pth            # Trained CNN model (not included)
├── hand_landmarker.task     # MediaPipe hand model (not included)
├── face_landmarker.task     # MediaPipe face model (not included)
├── templates/
│   └── index.html           # Full web UI (menu, training, practice)
├── static/
│   └── signs/               # Hand sign images (optional)
│       ├── horse.jpg
│       └── ...
└── sounds/                  # WAV sound effects (optional, auto-generated)
    ├── fireball.wav
    └── ...
```

---

## Controls

| Key | Action |
|-----|--------|
| `R` | Reset current jutsu / buffer |
| `G` | Toggle jutsu guide panel (Practice mode) |
| `Esc` | Return to main menu |

### Two-hand control (while a jutsu is active)
| Gesture | Effect |
|---------|--------|
| Pinch in/out (non-effect hand) | Scale effect size down/up |
| Rotate wrist (non-effect hand) | Shift effect hue/colour |

---

## Sharing with Others

Since the webcam must be local, this app cannot be deployed to a cloud platform. To share your session:

**On your local network:**
Access via your local IP as described above.

**Over the internet:**
Use [ngrok](https://ngrok.com/) to create a temporary public URL:

```bash
pip install pyngrok
ngrok http 5000
```

---

## Notes

- The CNN model file `jutsu_cnn.pth` is not included in this repository as it must be trained on your own hand data for best accuracy.
- The MediaPipe `.task` files are not included due to size — download links are in the Installation section.
- Performance is best with a well-lit environment and a clear background.

---

## Built With

- [Flask](https://flask.palletsprojects.com/) — web server and MJPEG streaming
- [MediaPipe](https://mediapipe.dev/) — hand and face landmark detection
- [PyTorch](https://pytorch.org/) — CNN hand sign classifier
- [OpenCV](https://opencv.org/) — camera capture and frame rendering
- [Pygame](https://www.pygame.org/) — audio synthesis and playback
