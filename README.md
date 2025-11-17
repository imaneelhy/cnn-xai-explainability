# cnn-xai-explainability

Comparing three explainability methods for convolutional neural networks on CIFAR-10:
**Grad-CAM**, **Saliency Maps**, and **Integrated Gradients**.

A ResNet-18 model is trained on CIFAR-10, and explanations are generated for correctly
classified test images. We then measure how similar these explanation maps are using
correlation and overlap metrics, and visualize qualitative differences.

---

## 1. Project Overview

**Goal:**  
Investigate whether different explanation methods highlight similar image regions for a CNN classifier.

**Research question:**  
> Do Grad-CAM, Saliency Maps, and Integrated Gradients produce similar attribution maps for a ResNet-18 trained on CIFAR-10?

---

## 2. Methods

- **Dataset:** CIFAR-10 (10 classes, 32×32 RGB images)
- **Model:** ResNet-18 adapted for CIFAR-10
  - First conv layer changed to 3×3, stride 1
  - No initial max-pooling
- **Training:** 5 epochs with SGD (lr=0.1, momentum=0.9, weight decay=5e-4)  
  Achieved ~75% test accuracy (sufficient for explainability experiments).

### Explainability methods

- **Grad-CAM**
  - Uses gradients w.r.t. the last convolutional block (`layer4`) to weight feature maps.
  - Produces a coarse, class-specific heatmap over the image.

- **Saliency Maps**
  - Absolute value of the gradient of the class score w.r.t. the input pixels.
  - Highlights pixels where small changes most affect the score; often noisy and edge-like.

- **Integrated Gradients**
  - Path integral of gradients from a baseline image (all zeros in normalized space) to the input.
  - Produces smoother attributions than raw gradients and is less sensitive to local noise.

### Similarity metrics

For each correctly classified test image, we compute:

- **Pearson correlation** between normalized heatmaps (flattened to vectors).
- **IoU@20%** (Intersection-over-Union):
  - Threshold each map to keep only the **top 20%** most activated pixels.
  - Compute IoU between the resulting binary masks.

Evaluations are done on **100 correctly classified test images**.

---

## 3. Quantitative Results

| Pair                 | Pearson (mean ± std) | IoU@20% (mean ± std) |
|----------------------|----------------------|----------------------|
| Grad-CAM vs Saliency | 0.221 ± 0.192        | 0.189 ± 0.081        |
| Grad-CAM vs IG       | 0.151 ± 0.095        | 0.166 ± 0.042        |
| Saliency vs IG       | 0.352 ± 0.086        | 0.232 ± 0.034        |

**Observations:**

- Saliency Maps and Integrated Gradients show the highest agreement, both in correlation and IoU.
- Grad-CAM is less similar to both gradient-based methods, which matches the intuition that:
  - Grad-CAM works on **feature maps** and yields coarse, object-level blobs.
  - Saliency and Integrated Gradients operate directly on **input gradients**, capturing finer details.

---

## 4. Qualitative Examples

The notebook also visualizes explanations side-by-side:

- Original image + predicted class
- Grad-CAM heatmap
- Saliency heatmap
- Integrated Gradients heatmap

These plots show that:
- Grad-CAM focuses on broad regions over the main object (e.g., the ship hull).
- Saliency maps highlight sharp edges and high-frequency details, sometimes including background.
- Integrated Gradients produce smoother maps that combine object and some contextual pixels.


