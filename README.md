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




## Qualitative Examples

The figure below shows qualitative comparisons between the three explanation
methods on correctly classified CIFAR-10 images.

<img width="1150" height="304" alt="download (11)" src="https://github.com/user-attachments/assets/82969273-5dba-4bd9-b187-e72be79d26aa" />
<img width="1150" height="304" alt="download (12)" src="https://github.com/user-attachments/assets/771b64c9-1e0b-4552-9775-730397b29cf9" />
<img width="1150" height="304" alt="download (14)" src="https://github.com/user-attachments/assets/7bf1affb-0ef6-42eb-ab68-be10198d55de" />

Each row corresponds to one test image (e.g. *ship*, *truck*).  
From left to right:

1. **Original** – the input image with the model’s predicted class.
2. **Grad-CAM** – produces a smooth, blob-like heatmap that highlights large,
   semantically meaningful regions. In the examples above, Grad-CAM concentrates
   on the main object: the hull of the ship and the body of the truck.
3. **Saliency** – based on raw input gradients. These maps are much more
   fragmented and tend to emphasize high-frequency edges and small details,
   including some background pixels.
4. **Integrated Gradients** – smoother than raw saliency but still more
   fine-grained than Grad-CAM. Attributions are spread across important object
   parts (e.g. the front of the truck, the upper structure of the ship) while
   also assigning some weight to surrounding context.

Overall, the examples illustrate that:
- Grad-CAM gives a coarse, object-level view of what the network is “looking at”.
- Saliency Maps and Integrated Gradients reveal finer pixel-level structure,
  but can appear noisier.
- Different methods highlight overlapping but not identical regions, which
  matches the quantitative similarity scores reported above.


