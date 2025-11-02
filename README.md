# 🚗 Autonomous Car Platform — Lane and Traffic Sign Detection (Work in Progress)

**Status:** Development in progress  
**Language:** Python 3.13.7  
**Operating System:** Windows 11  

This repository contains the core vision algorithms for an **autonomous car platform** that will eventually operate as a small, physical vehicle.  
The main objective of this project is to develop a system capable of **independent movement along a simulated road**, detecting and reacting to **traffic signs** from a predefined image database.

The project is developed as part of an **engineering thesis at Wrocław University of Science and Technology**.

---

## 🎯 Project Overview

The platform aims to:
- Follow road lanes based on camera vision,
- Detect and classify road signs from a live camera feed,
- Eventually integrate both systems on a **Raspberry Pi controller** for autonomous driving.

Currently, two independent modules are implemented:

| Module | Description |
|--------|--------------|
| **`Line_detect.py`** | Detects the center of a driving lane on a road image using perspective transform and histogram analysis. |
| **`Sign_detect.py`** | Recognizes road signs from a live camera feed using classical computer vision (ORB descriptors, color & shape analysis). |

Both algorithms work **without machine learning** — the system relies solely on deterministic image processing methods.

---

## ⚙️ Features (Current State)

### 🧩 Lane Detection (`Line_detect.py`)
- Works on static road images (e.g., `Linia_drogi/droga2.png`)
- Uses **HSV thresholding** and morphological operations to isolate lane markings
- Applies **perspective transform** to simulate a top-down view
- Computes lane center using **histogram-based analysis**
- Interactive configuration via **trackbars** for ROI and color thresholds

### 🚸 Traffic Sign Detection (`Sign_detect.py`)
- Real-time video capture from laptop camera (via OpenCV)
- Multi-stage detection combining:
  1. **Edge and contour extraction**
  2. **Shape classification** (triangle, circle, octagon)
  3. **Dominant color detection** (HSV-based)
  4. **ORB descriptor matching** against an image database (`baza_do_porownania`)
- Classifies signs such as:
  - Stop  
  - Warning (triangular, orange)  
  - Speed limits  
  - Mandatory signs (blue circular)

---

## 🧠 Technologies Used

- **Python 3.13.7**
- **OpenCV**
- **NumPy**
- **Math**
- **Jupyter Notebook** (for prototyping and testing)

> No machine learning or deep learning techniques are used in this project.

---

## 🗂️ Repository Structure

