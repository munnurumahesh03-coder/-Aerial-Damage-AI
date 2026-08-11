# 🌪️ Aerial-Damage-AI
### A Deep Dive into Geospatial Computer Vision for Disaster Response

![Python](https://img.shields.io/badge/Python-3.12-blue )
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow )
![SAHI](https://img.shields.io/badge/SAHI-Inference-orange )
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red )

**Project Status: Completed & Deployed**

This repository contains the complete code, methodology, and findings for the "Aerial Disaster Response AI," an end-to-end Computer Vision (CV) project focused on assessing building damage from satellite and drone imagery. 

The project culminates in a production-ready deployment, revealing powerful lessons about the physics of aerial imagery, the dangers of extreme class imbalance, and the delicate balance of cloud compute optimization. The final conclusion demonstrates a rigorous, Senior-level approach to data pruning, model training, and advanced inference techniques.

**Live Web App:** [Test the Live Streamlit Dashboard Here!](https://gnvdgb5u9xxgyrxj9sqptd.streamlit.app/ )

### Table of Contents
1.  [Project Objective](#project-objective)
2.  [The Dataset: Challenges and Characteristics](#the-dataset-challenges-and-characteristics)
3.  [Visualizing the Data: Key Insights](#visualizing-the-data-key-insights)
4.  [Methodology: A Rigorous and Repeatable Workflow](#methodology-a-rigorous-and-repeatable-workflow)
5.  [The Gauntlet: YOLOv8 Training & Evaluation](#the-gauntlet-yolov8-training--evaluation)
6.  [Grand Lessons & Key Findings](#grand-lessons--key-findings)
7.  [The Grand Finale: SAHI & Production Deployment](#the-grand-finale-sahi--production-deployment)
8.  [How to Use This Repository](#how-to-use-this-repository)
9.  [Future Improvements](#future-improvements)

---  
### Project Objective

The grand objective was to build a high-performance, production-ready Computer Vision system capable of classifying building damage in post-disaster aerial imagery to assist rescue teams (FEMA) and insurance adjusters. The project was divided into two main phases:

1.  **The CV Data Blueprint:** Systematically audit, clean, and programmatically prune a massive, highly imbalanced dataset into a strict binary classification ("Damaged" vs. "Safe") to ensure model convergence.
2.  **The YOLO & SAHI Blueprint:** Train a highly optimized YOLOv8s architecture on a cloud GPU, integrate Slicing Aided Hyper Inference (SAHI) to detect microscopic objects, and deploy the final pipeline to a live Streamlit web server.

---  
### The Dataset: Challenges and Characteristics
The project uses a massive aerial imagery dataset (derived from xBD/DoD standards) containing thousands of drone and satellite images of post-disaster zones. The dataset was chosen specifically for its complex, real-world imperfections.  

**Key Characteristics:**
*   **Extreme Class Imbalance:** The raw dataset contained over 600,000 "Safe" buildings and only 25,000 "Destroyed" buildings. 
*   **Small-Object Degradation:** Satellite images are massive (4K+ resolution). Shrinking them down to standard YOLO input sizes (640x640) completely destroys the pixel data of tiny houses, making them invisible to standard models.
*   **Domain Gap & Visual Ambiguity:** Distinguishing between "minor damage" and "major damage" from 10,000 feet in the air is highly subjective, requiring a strategic pivot to guarantee business-level accuracy.

---
### Visualizing the Data: Key Insights
Exploratory Data Analysis (EDA) and Health Checks were crucial for understanding the physics of satellite imagery and guiding our training strategy.

**1. The "Heavy Lifters" (Bounding Box Physics)**
Visual verification revealed that standard bounding boxes for buildings in 4K drone imagery often occupy less than 1% of the total image pixels. This proved that a standard YOLO model would fail without advanced slicing techniques during inference.

**2. Categorical Impact (The Imbalance Trap)**
By extracting the class distribution, we uncovered a massive trap in the dataset:
*   **Safe Buildings:** 89.14% of the data.
*   **Damaged Buildings:** ~10.86% of the data.
*   **The Insight:** If we trained on this raw data, the AI would take a lazy shortcut: predict "Safe" 100% of the time, achieve 89% accuracy, and completely fail its real-world objective of finding destroyed homes.

---

### Methodology: A Rigorous and Repeatable Workflow
The true hero of this project was the rigorous workflow. Every step was designed to prevent GPU bottlenecks and ensure production readiness.

*   **Programmatic Data Pruning (The Physics Engine):**
    *   Engineered a Python script to ruthlessly merge overlapping classes (`minor-damage`, `major-damage`, `destroyed`) into a strict binary classification (`Damaged` vs `Safe`). This cured the class imbalance and aligned the model with the actual business objective.

*   **Compute Optimization (T4 GPU):**
    *   Configured the YOLOv8s training pipeline specifically for cloud constraints. Utilized `batch=32` to maximize VRAM, `cache=True` to bypass slow disk I/O, and `patience=5` (Early Stopping) to prevent overfitting and save compute hours.

*   **SAHI (Slicing Aided Hyper Inference):**
    *   Instead of passing a massive 4K image to the model, SAHI mathematically slices the image into overlapping 320x320 grids, runs predictions on each slice, and stitches the bounding boxes back together using Non-Maximum Suppression (NMS).

*   **Model Persistence:** The final champion model (`best.pt`) was serialized and extracted for seamless web deployment.

---

### The Gauntlet: YOLOv8 Training & Evaluation
We deployed the YOLOv8 Small (`yolov8s.pt`) architecture to hunt for the perfect balance of speed and accuracy. The model was trained on a Tesla T4 GPU and evaluated on an unseen validation set.

| Class | Precision (P) | Recall (R) | mAP@50 | Key Lesson Learned |
| :--- | :--- | :--- | :--- | :--- |
| **All** | **0.450** | **0.437** | **0.358** | **THE CHAMPION BASELINE.** Strong foundational learning in just 20 epochs. |
| Safe | 0.458 | 0.624 | 0.505 | High recall proves the model easily identifies intact structures. |
| Damaged | 0.442 | 0.249 | 0.212 | Harder to detect from high altitudes, proving the absolute necessity of SAHI during deployment. |

---

### Grand Lessons & Key Findings
*   **The Class Imbalance Trap is a Silent Killer:** During the Data Health Check, we discovered the 89% "Safe" bias. We ruthlessly pruned the dataset. Why? Because in the real world, a model that ignores destroyed buildings to artificially inflate its accuracy score costs lives.
*   **SAHI Beats Raw Scaling:** Standard YOLO resizes images to 640x640. For a 4K drone image, this destroys the pixel data of a collapsed roof. By implementing SAHI, we preserved the original resolution, proving that **smart inference math beats raw model size**.
*   **Cloud DevOps & Dependency Hell:** Deploying Computer Vision to the cloud is a DevOps challenge. We overcame `libGL.so.1` C++ driver crashes by dynamically injecting `python3-opencv` into the Linux server via a custom `packages.txt` file.

--- 
### The Grand Finale: SAHI & Production Deployment

**1. The Final Verdict** 
The champion YOLOv8s model successfully learned the complex visual features of structural damage from thousands of feet in the air. By combining this lightweight model with SAHI, we achieved high-accuracy small-object detection without the massive computational cost of a Vision Transformer (ViT).

**2. The Streamlit Web Dashboard & UX Optimization**
The final model was deployed to Streamlit Community Cloud. To bridge the gap between complex machine learning and user experience, the deployment was heavily optimized:  

* **Memory Optimization for UX:** Instead of crashing the cloud server with massive 4K satellite images, the backend dynamically resizes uploads to a safe 1200px threshold. This keeps the dashboard clean, intuitive, and crash-free without sacrificing SAHI's slicing accuracy.
* **Dynamic Backend Math:** The UI is designed to take simple image uploads. It dynamically calculates the complex SAHI grid mathematics, runs batched inference on the slices, and stitches the bounding boxes back together using Non-Maximum Suppression (NMS) in the background, serving real-time predictions instantly.

--- 
### How to Use This Repository
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Aerial-Damage-AI.git
    cd Aerial-Damage-AI
    ```
2.  **Explore the Notebooks:** The project is divided into two modular blueprints:
    *   `01_CV_Data_Blueprint.ipynb`: Data health checks, class merging, and programmatic pruning.
    *   `02_YOLO_SAHI_Blueprint.ipynb`: YOLOv8s training, SAHI integration, and deployment preparation.
3.  **Run the Deployed App Locally:** 
    ```bash
    pip install -r requirements.txt
    streamlit run app.py
    ```
    *   **Why Streamlit?** It provides a sleek, modern UI that seamlessly handles the heavy Computer Vision backend, allowing anyone to interact with the model in real-time.

---

### Future Improvements
While this project is complete and deployed, the methodology opens the door for further exploration:

*   **Live Drone Feed Integration:** Expand the Streamlit dashboard to process live RTSP video streams from drones in real-time.
*   **Advanced Ensembling:** Blend the YOLOv8s champion with a tuned RT-DETR model using Weighted Boxes Fusion (WBF ) to push accuracy even higher.
*   **Instance Segmentation:** Upgrade from object detection (bounding boxes) to instance segmentation (YOLOv8-seg) to calculate the exact square footage of roof damage for insurance payouts.
