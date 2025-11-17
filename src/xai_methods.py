import torch
import torch.nn.functional as F

# ---------------- Grad-CAM ---------------- #


class GradCAM:
    """
    Minimal Grad-CAM implementation.

    Usage:
        grad_cam = GradCAM(model, target_layer_name="layer4")
        cam = grad_cam.generate(input_tensor, target_class)  # (1, Hc, Wc)
    """

    def __init__(self, model, target_layer_name: str):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer_name]

        self.activations = None
        self.gradients = None

        self.fwd_hook = self.target_layer.register_forward_hook(self._forward_hook)
        # modern API vs deprecated register_backward_hook
        self.bwd_hook = self.target_layer.register_full_backward_hook(
            self._backward_hook
        )

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        # grad_output is a tuple; take gradients w.r.t. activations
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int):
        """
        input_tensor: (1, C, H, W)
        returns: normalized Grad-CAM map (1, Hc, Wc) in [0, 1]
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        score = logits[:, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients  # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        weights = grads.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam  # (1, 1, H, W)

    def remove_hooks(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()


# ---------------- Saliency Maps ---------------- #


def compute_saliency(model, input_tensor: torch.Tensor, target_class: int):
    """
    Returns normalized saliency map (H, W) in [0, 1].
    """
    x = input_tensor.clone().detach()
    x.requires_grad_(True)

    model.zero_grad()
    logits = model(x)
    score = logits[:, target_class]
    score.backward()

    gradient = x.grad.data  # (1, C, H, W)
    saliency = gradient.abs().max(dim=1)[0].squeeze(0)  # (H, W)

    saliency -= saliency.min()
    saliency /= (saliency.max() + 1e-8)
    return saliency


# ---------------- Integrated Gradients ---------------- #


def integrated_gradients(
    model,
    input_tensor: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    steps: int = 50,
):
    """
    Integrated Gradients for a single input.
    Returns normalized attribution map (H, W) in [0, 1].
    """
    device = input_tensor.device
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)

    scaled_inputs = [
        baseline + (float(i) / steps) * (input_tensor - baseline)
        for i in range(1, steps + 1)
    ]
    scaled_inputs = torch.cat(scaled_inputs, dim=0)
    scaled_inputs.requires_grad_(True)

    model.zero_grad()
    logits = model(scaled_inputs)
    target_scores = logits[:, target_class].sum()
    target_scores.backward()

    grads = scaled_inputs.grad  # (steps, C, H, W)
    avg_grads = grads.mean(dim=0, keepdim=True)  # (1, C, H, W)

    attributions = (input_tensor - baseline) * avg_grads  # (1, C, H, W)
    attributions = attributions.sum(dim=1).squeeze(0)  # (H, W)
    attributions = attributions.clamp(min=0)

    attributions -= attributions.min()
    attributions /= (attributions.max() + 1e-8)
    return attributions


# ---------------- Metrics & Utils ---------------- #


def flatten_and_normalize(map1, map2):
    m1 = map1.float().view(-1)
    m2 = map2.float().view(-1)
    m1 = (m1 - m1.min()) / (m1.max() - m1.min() + 1e-8)
    m2 = (m2 - m2.min()) / (m2.max() - m2.min() + 1e-8)
    return m1, m2


def pearson_correlation(map1, map2):
    m1, m2 = flatten_and_normalize(map1, map2)
    m1 = m1 - m1.mean()
    m2 = m2 - m2.mean()
    num = (m1 * m2).sum()
    den = (m1.norm() * m2.norm() + 1e-8)
    return (num / den).item()


def iou_topk(map1, map2, k: float = 0.2):
    """
    IoU between top-k fraction of pixels (k in (0,1]).
    """
    m1, m2 = flatten_and_normalize(map1, map2)
    n = m1.numel()
    topk = int(max(1, k * n))

    thresh1 = torch.topk(m1, topk).values.min()
    thresh2 = torch.topk(m2, topk).values.min()

    mask1 = m1 >= thresh1
    mask2 = m2 >= thresh2

    inter = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float() + 1e-8
    return (inter / union).item()


def denormalize(img_tensor, mean, std):
    """
    Undo normalization for visualization.
    """
    mean = torch.tensor(mean).view(3, 1, 1).to(img_tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(img_tensor.device)
    return img_tensor * std + mean
