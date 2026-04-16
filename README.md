# DS50 Project - Car Face Authentication System

## Project Overview
This project is part of the DS50 teaching unit at UTBM and aims to implement a **local, 1-to-1 facial authentication system** for identifying a car driver using a camera. The goal is to ensure fast and reliable recognition without relying on cloud services, suitable for deployment in an embedded context such as a car.

### Key Constraints
- Must run **entirely offline** (no cloud)
- Supports a **limited user base** (initially 1, up to ~20)
- Handles **real-world conditions**: varying lighting, camera angles, etc.
- Embedding base should **evolve over time** to adapt to changes in the user's appearance

## Project Structure
```
DS50-CAR-FACE-AUTH/
├── data/ # Directory for storing images and embeddings
├── notebooks/ # Jupyter notebooks for experimentation
├── src/ # Source code for the project
│   ├── antispoofing/ # Anti-spoofing module
│   ├── __init__.py # Package initialization
│   ├── config.py # Configuration file for model paths and parameters
│   ├── database.py # Embedding database management (save/load/update)
│   ├── detection.py # Face detection and alignment
│   ├── embedding.py # Face embedding generation (using pre-trained model)
│   ├── recognition.py # Matching logic (comparison with stored embeddings)
│   ├── gui.py # Graphical user interface for the application
│   ├── main.py # Main entry point for the application
│   ├── enroll_face.py # Enrollment process for adding new faces to the database
│   └── utils.py # Utility functions (e.g., image processing)
├── requirements.txt # Python package dependencies
└── README.md # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.12 or higher

### 1. Create a virtual environment
```
python -m venv venv
source venv/Scripts/activate  # Windows
source venv/bin/activate  # Linux/MacOS
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

### 3. Download the pre-trained models
>Note: the required models are already included in the submitted archive. Otherwise, you can download them from the provided links below.

Download the pre-trained models (onnx files) from the following links and place them in the `models/` directory: https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing

For the anti-spoofing model (pytorch), you can find it [here](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/b6d5f04ad78778917853b25c778acef6d5626d15/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth).

### 4. Run the application
```
python -m src.main
```

## Usage
- **Enrollment**: Use the `enroll_face.py` script to add a new face to the database. Follow the prompts to capture images and generate embeddings. To run the script, execute:
  ```
  python -m src.enroll_face
  ```
- **Recognition**: The main application will automatically detect faces and attempt to recognize them using the stored embeddings. If a match is found, the corresponding name will be displayed.

## Core features
- Local and offline recognition: Entirely self-contained, no cloud or network connection needed.
- One-to-one authentication: Verifies identity against stored reference.
- Robust face detection & alignment: Uses SCRFD-10GF for accurate detection and normalization.
- Face embedding with ArcFace loss: Powered by ResNet50 trained on WebFace600K for high precision.
- Anti-spoofing protection: Integration of MiniFASNetV2 model to detect fake faces (photos, videos, masks).
- On-device continuous adaptation: The system evolves over time using exponential moving average of embeddings.
- Graphical user interface: Lightweight GUI for real-time visualization and testing.
- Easy face enrollment: Scripted process to register new users quickly via webcam.
- Embeddable design: Built for deployment on resource-constrained devices like in-car systems.

## Benchmarking
To facilitate testing and performance evaluation, we provide:
- A **ready-to-use benchmark notebook** located in the `notebooks/` folder.
- A **sample image set** located under `data/test/` and `data/references/`.

The notebook allows you to evaluate the recognition system’s accuracy, speed, and robustness under various conditions using reference and test images organized by identity. This makes it easy to reproduce experiments or validate improvements.

## Credits
This project was developed as part of the DS50 unit at UTBM in Spring 2025 by the following students (FISE, Data Science):
- Léonard Zipper
- Gabriel Fleuret
- Quentin Balezeau
- Chengjie Yang
- Estouan Gachelin
- Benoît Brindejonc

Supervised by: Sid Ahmed Lamrous

## Third-Party Credits
This project builds on and integrates the following  open-source and research tools:
- [InsightFace](https://github.com/deepinsight/insightface)
  - Used for face detection (SCRFD-10GF) and embeddings (ResNet50@WebFace600K).
- [Silent-Face Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing)
  - Used for liveness detection.
  - Some files in src/antispoofing/ are derived from this project and properly credited in their headers.

If you have any questions or encounter any issues while reviewing or running the project, feel free to contact us.