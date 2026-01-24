# Face Detection + Face Recognition 


---

## Approach 1: 
**Best for:** Small, static datasets where a pre-defined classifier is sufficient.

* **Detection:** `MTCNN` (Landmark-based detection and alignment)
* **Embedding:** `FaceNet` (Generates a 128-d feature vector)
* **Recognition:** `SVM Classifier` (Trained to categorize faces into specific classes)



---

## Approach 2: Optimal
**Best for:** Real-time applications and scalable "open-set" recognition.

* **Detection:** `YOLOv8-Face` (High-speed, single-stage detection)
* **Embedding:** `ArcFace` (Generates a 512-d angular margin embedding)
* **Recognition:** `Cosine Similarity` (Mathematical matching; handles new users without retraining)



---

