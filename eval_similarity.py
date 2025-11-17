import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.model import (
    get_resnet18_cifar10,
    get_cifar10_loaders,
    DEVICE,
    CIFAR10_MEAN,
    CIFAR10_STD,
)
from src.xai_methods import (
    GradCAM,
    compute_saliency,
    integrated_gradients,
    pearson_correlation,
    iou_topk,
    denormalize,
)


CHECKPOINT_PATH = "resnet18_cifar10.pth"  # or "checkpoints/resnet18_cifar10.pth"


def load_model():
    model = get_resnet18_cifar10().to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
    else:
        print(
            f"WARNING: checkpoint {CHECKPOINT_PATH} not found. "
            "Train the model in the notebook and save it first."
        )
    model.eval()
    return model


def evaluate_explanations(model, explain_loader, num_max=100, k=0.2):
    grad_cam = GradCAM(model, target_layer_name="layer4")

    pearson_scores = {"gc_sal": [], "gc_ig": [], "sal_ig": []}
    iou_scores = {"gc_sal": [], "gc_ig": [], "sal_ig": []}

    n_explained = 0

    for img, label in explain_loader:
        if n_explained >= num_max:
            break

        img, label = img.to(DEVICE), label.to(DEVICE)

        # prediction
        with torch.no_grad():
            logits = model(img)
            pred = logits.argmax(dim=1)

        # only use correctly classified examples
        if pred.item() != label.item():
            continue

        x = img  # (1, 3, 32, 32)
        target_class = pred.item()

        # Grad-CAM
        cam = grad_cam.generate(x, target_class)  # (1, 1, Hc, Wc)
        cam_up = F.interpolate(
            cam, size=(32, 32), mode="bilinear", align_corners=False
        ).squeeze()  # (H, W)

        # Saliency
        sal = compute_saliency(model, x, target_class)  # (H, W)

        # Integrated Gradients
        ig = integrated_gradients(model, x, target_class, steps=32)  # (H, W)

        # metrics
        pearson_scores["gc_sal"].append(pearson_correlation(cam_up, sal))
        pearson_scores["gc_ig"].append(pearson_correlation(cam_up, ig))
        pearson_scores["sal_ig"].append(pearson_correlation(sal, ig))

        iou_scores["gc_sal"].append(iou_topk(cam_up, sal, k=k))
        iou_scores["gc_ig"].append(iou_topk(cam_up, ig, k=k))
        iou_scores["sal_ig"].append(iou_topk(sal, ig, k=k))

        n_explained += 1

    grad_cam.remove_hooks()

    print(f"Number of explained images used: {n_explained}")

    # print mean ± std for each pair
    for key in pearson_scores:
        arr_p = np.array(pearson_scores[key])
        arr_i = np.array(iou_scores[key])
        print(
            f"{key} -> Pearson: {arr_p.mean():.3f} ± {arr_p.std():.3f}, "
            f"IoU@{int(k*100)}%: {arr_i.mean():.3f} ± {arr_i.std():.3f}"
        )

    return pearson_scores, iou_scores


if __name__ == "__main__":
    # get loaders (only need explain_loader here)
    _, _, explain_loader, classes = get_cifar10_loaders()

    # load trained model
    model = load_model()

    # run evaluation
    evaluate_explanations(model, explain_loader, num_max=100, k=0.2)
