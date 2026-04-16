# Implementation Guide: Car Face Authentication System

This document outlines the transition from a desktop Python prototype to a production-ready, embedded automotive system.

## 1. Hardware Requirements

### Compute Unit (The "Brain")
A standard laptop is not suitable for automotive environments. You need an embedded edge AI device:
- **High Performance:** NVIDIA Jetson Orin Nano / NX (Best for AI, supports TensorRT).
- **Budget/Prototyping:** Raspberry Pi 5 (Good for testing, but slower for complex models).
- **Automotive Grade:** NXP i.MX 8 or Qualcomm Snapdragon Automotive (Industry standard).

### Camera System
- **Placement:** Dashboard or steering column mount.
- **Sensor:** A High-Dynamic Range (HDR) camera to handle varying light.
- **Infrared (NIR):** Highly recommended for night-time operation. *Note: Using NIR requires retraining or swapping the current RGB-based models.*

### Vehicle Integration
- **CAN Bus Module:** (e.g., PiCAN for Raspberry Pi) to communicate with the car's Electronic Control Units (ECU).
- **Power Supply:** 12V/24V to 5V automotive-grade step-down converter (to handle voltage spikes and prevent battery drain).

---

## 2. Software & Architecture Adjustments

### Remove Graphical Interface
A real car system runs "headless" (no monitor).
- **Task:** Modify `src/main.py` and `src/gui.py` to remove `cv2.imshow()` and `cv2.waitKey()`.
- **Replacement:** Use GPIO pins or CAN bus messages to provide feedback (e.g., a green LED for success, a red LED for failure).

### Model Optimization
The current models are heavy for embedded CPUs.
- **Quantization:** Convert `.onnx` and `.pth` models to FP16 or INT8 formats.
- **Acceleration:** Use **TensorRT** (NVIDIA), **OpenVINO** (Intel), or **TFLite** to achieve 30+ FPS.

### Security & Encryption
- **Data Protection:** Encrypt the `data/` folder. If a thief steals the SD card, they should not be able to modify the reference embeddings.
- **Secure Boot:** Ensure the embedded OS only runs signed, trusted code.

---

## 3. Step-by-Step Implementation Plan

### Phase 1: Embedded Prototyping (Bench Testing)
1. **Setup:** Install the project on an edge device (e.g., Jetson Nano).
2. **Benchmark:** Measure the current FPS. If it's below 10 FPS, optimize the models immediately.
3. **Trigger Logic:** Replace keyboard inputs ('s' for save, 'q' for quit) with physical buttons connected to GPIO pins.

### Phase 2: Headless Service Integration
1. **Automation:** Create a Linux `systemd` service to start the authentication script automatically on boot.
2. **Persistence:** Ensure the `database.py` saves changes immediately to prevent data loss during sudden power-offs.
3. **Error Handling:** Implement robust logging that doesn't fill up the disk space (rotate logs).

### Phase 3: Vehicle Integration
1. **CAN Bus Integration:** Write a Python bridge using `python-can` to send the "Unlock Engine" or "Adjust Seat" commands upon successful authentication.
2. **Physical Install:** Securely mount the camera to avoid vibration-induced blur.
3. **Testing:** Test in extreme conditions (direct sunlight, pitch black, driver wearing sunglasses/hat).

---

## 4. Key Logic Changes Needed

| Component | Current State | Production State |
| :--- | :--- | :--- |
| **Input** | Webcam (USB) | HDR/NIR Global Shutter Camera |
| **Output** | OpenCV Window | CAN Bus / GPIO / Infotainment Screen |
| **Enrollment** | `enroll_face.py` (CLI) | Dedicated "New Driver" mode in Car Settings |
| **Power** | AC Adapter | 12V Car Battery (with Sleep/Wake logic) |
