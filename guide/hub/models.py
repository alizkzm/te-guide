"""The model hub: 26 ImageNet backbones + the layer surgery every score needs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tvm

from guide import config

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


CNN_SPECS = {
    "googlenet":       (tvm.googlenet,       tvm.GoogLeNet_Weights,       {"aux_logits": True}),
    "mobilenet_v2":    (tvm.mobilenet_v2,    tvm.MobileNet_V2_Weights,    {}),
    "mnasnet1_0":      (tvm.mnasnet1_0,      tvm.MNASNet1_0_Weights,      {}),
    "densenet121":     (tvm.densenet121,     tvm.DenseNet121_Weights,     {}),
    "densenet161":     (tvm.densenet161,     tvm.DenseNet161_Weights,     {}),
    "densenet169":     (tvm.densenet169,     tvm.DenseNet169_Weights,     {}),
    "densenet201":     (tvm.densenet201,     tvm.DenseNet201_Weights,     {}),
    "efficientnet_b0": (tvm.efficientnet_b0, tvm.EfficientNet_B0_Weights, {}),
    "resnet18":        (tvm.resnet18,        tvm.ResNet18_Weights,        {}),
    "resnet34":        (tvm.resnet34,        tvm.ResNet34_Weights,        {}),
    "resnet50":        (tvm.resnet50,        tvm.ResNet50_Weights,        {}),
    "resnet101":       (tvm.resnet101,       tvm.ResNet101_Weights,       {}),
    "resnet152":       (tvm.resnet152,       tvm.ResNet152_Weights,       {}),
    "inception_v3":    (tvm.inception_v3,    tvm.Inception_V3_Weights,    {"aux_logits": True}),
}
CNN_BUILDERS = CNN_SPECS


def _weights_enum(model_name: str, version: str):
    """`"v1"` -> IMAGENET1K_V1; `"default"` -> torchvision's DEFAULT (V2 where it exists)."""
    enum = CNN_SPECS[model_name][1]
    if str(version).lower() in ("default", "v2", "best"):
        return enum.DEFAULT
    return getattr(enum, "IMAGENET1K_V1", enum.DEFAULT)


def weights_tag(model_name: str, version: str) -> str:
    """The checkpoint actually used, recorded in the probe metadata."""
    name = model_name.lower()
    if name in ITM_EFFICIENTNET:
        return "efficientnet_pytorch"
    if name not in CNN_SPECS:
        return "timm-pretrained" if name in VIT_BUILDERS else "external-checkpoint"
    return str(_weights_enum(name, version)).split(".")[-1]


SOURCE_TOP1 = {
    "googlenet": 69.778,
    "mobilenet_v2": 71.878,
    "mnasnet1_0": 73.456,
    "densenet121": 74.434,
    "densenet161": 77.138,
    "densenet169": 75.600,
    "densenet201": 76.896,
    "efficientnet_b0": 77.692,
    "resnet18": 69.758,
    "resnet34": 73.314,
    "resnet50": 76.130,
    "resnet101": 77.374,
    "resnet152": 78.312,
    "inception_v3": 77.294,
    "vit_t_16": 75.45,
    "vit_s_16": 81.39,
    "vit_b_16": 84.53,
    "pvt_t": 70.5,
    "pvt_s": 78.7,
    "pvt_m": 81.2,
    "pvtv2_b2": 82.0,
    "swin_t": 81.18,
}

SOURCE_TOP1_V2 = {
    "resnet50": 80.858,
    "resnet101": 81.886,
    "resnet152": 82.284,
    "mobilenet_v2": 72.154,
}

IMAGE_SIZE_OVERRIDES = {"inception_v3": 299, "dino_vits8": 224}

SSL_VIT_KEYS = {
    "mae_vitb16",
    "mae_vitl16",
    "dino_vitb16",
    "dino_vits8",
    "dino_vits16",
    "mocov3_vitb16",
    "mocov3_vits16",
    "simmim_vitb16",
}

VIT_BUILDERS = {
    "mae_vitb16": ["vit_base_patch16_224.mae"],
    "mae_vitl16": ["vit_large_patch16_224.mae"],
    "dino_vitb16": ["vit_base_patch16_224.dino"],
    "dino_vits8": ["vit_small_patch8_224.dino"],
    "dino_vits16": ["vit_small_patch16_224.dino"],
    "mocov3_vitb16": ["vit_base_patch16_224.mocov3"],
    "mocov3_vits16": ["vit_small_patch16_224.mocov3"],
    "simmim_vitb16": ["vit_base_patch16_224.simmim"],
    "vit_t_16": ["vit_tiny_patch16_224.augreg_in21k_ft_in1k", "vit_tiny_patch16_224"],
    "vit_s_16": ["vit_small_patch16_224.augreg_in21k_ft_in1k", "vit_small_patch16_224"],
    "vit_b_16": ["vit_base_patch16_224.augreg_in21k_ft_in1k", "vit_base_patch16_224"],
    "pvtv2_b2": ["pvt_v2_b2"],
    "pvt_t": ["pvt_v2_b0", "pvt_tiny"],
    "pvt_s": ["pvt_v2_b1", "pvt_small"],
    "pvt_m": ["pvt_v2_b3", "pvt_medium"],
    "swin_t": ["swin_tiny_patch4_window7_224.ms_in1k", "swin_tiny_patch4_window7_224"],
}

SSL_CNN_MODELS = [
    "byol", "deepclusterv2", "infomin", "insdis", "mocov1", "mocov2",
    "pclv1", "pclv2", "selav2", "simclrv1", "simclrv2", "swav",
]

SSL_CNN_CHECKPOINT_DIR = config.PKG_ROOT / "checkpoints" / "ssl_cnn"


ITM_EFFICIENTNET = {"efficientnet_b0": "efficientnet-b0"}


def _load_efficientnet_pytorch(tag: str) -> nn.Module:
    try:
        from efficientnet_pytorch import EfficientNet
    except ImportError as exc:
        raise ImportError(
            "efficientnet_b0 follows ITM's base_model.py, which uses the "
            "`efficientnet_pytorch` package rather than torchvision.  "
            "Install it with `pip install efficientnet_pytorch`."
        ) from exc
    return EfficientNet.from_pretrained(tag).eval()


def _ssl_search_dirs() -> list:
    """Where to look for SSL-CNN checkpoints, in priority order: $GUIDE_SSL_CKPT, then checkpoints/ssl_cnn/, then checkpoints/."""
    cands = []
    env = os.environ.get("GUIDE_SSL_CKPT")
    if env:
        cands.append(Path(env))
    cands += [SSL_CNN_CHECKPOINT_DIR, config.PKG_ROOT / "checkpoints"]
    return [d for d in cands if d.exists()]


def _ssl_checkpoint_path(name: str) -> Path:
    """Locate the checkpoint for `name`, tolerant of both directory and separator style: the SFDA release names its files with hyphens (`moco-v1.pth`) and may sit directly in `checkpoints/` rather than `checkpoints/ssl_cnn/`."""
    key = name.replace("-", "").replace("_", "").lower()
    for d in _ssl_search_dirs():
        exact = d / f"{name}.pth"
        if exact.exists():
            return exact
        for f in d.glob("*.pth"):
            if f.stem.replace("-", "").replace("_", "").lower() == key:
                return f
    return SSL_CNN_CHECKPOINT_DIR / f"{name}.pth"


def _load_ssl_cnn(name: str) -> nn.Module:
    """ResNet-50 trunk + a self-supervised checkpoint (head-less)."""
    path = _ssl_checkpoint_path(name)
    if not path.exists():
        searched = ", ".join(str(d) for d in _ssl_search_dirs()) or "(none exist)"
        raise FileNotFoundError(
            f"self-supervised CNN '{name}' needs a checkpoint (e.g. {name}.pth or "
            f"a hyphen/underscore variant).\n    Searched: {searched}\n"
            f"    Set GUIDE_SSL_CKPT to the folder holding the .pth files, or place "
            f"them in checkpoints/ssl_cnn/."
        )
    model = tvm.resnet50(weights=None)
    model.fc = nn.Identity()
    state = torch.load(path, map_location="cpu")
    state = state.get("state_dict", state.get("model", state))
    state = {k.replace("module.", "").replace("encoder_q.", "").replace("backbone.", ""): v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(missing) > 20:
        raise RuntimeError(
            f"checkpoint {path} does not look like a ResNet-50 trunk "
            f"({len(missing)} missing keys)"
        )
    return model.eval()


SSL_VIT_LOCAL = {
    "mocov3_vitb16": "vit_base_patch16_224",
    "mocov3_vits16": "vit_small_patch16_224",
    "simmim_vitb16": "vit_base_patch16_224",
}

SSL_VIT_CHECKPOINT_DIR = config.PKG_ROOT / "checkpoints" / "vit"


def _ssl_vit_search_dirs() -> list:
    """Where to look for SSL-ViT checkpoints: $GUIDE_VIT_CKPT, checkpoints/vit/, then checkpoints/."""
    cands = []
    env = os.environ.get("GUIDE_VIT_CKPT")
    if env:
        cands.append(Path(env))
    cands += [SSL_VIT_CHECKPOINT_DIR, config.PKG_ROOT / "checkpoints"]
    return [d for d in cands if d.exists()]


def _ssl_vit_checkpoint_path(name: str) -> Optional[Path]:
    """Locate a checkpoint for `name`, tolerant of extension (.pth / .pth.tar) and hyphen/underscore style."""
    key = name.replace("-", "").replace("_", "").lower()
    for d in _ssl_vit_search_dirs():
        for ext in (".pth", ".pth.tar"):
            exact = d / f"{name}{ext}"
            if exact.exists():
                return exact
        for f in list(d.glob("*.pth")) + list(d.glob("*.pth.tar")):
            stem = f.name
            for ext in (".pth.tar", ".pth"):
                if stem.endswith(ext):
                    stem = stem[: -len(ext)]
                    break
            if stem.replace("-", "").replace("_", "").lower() == key:
                return f
    return None


def _remap_ssl_vit_state(state: dict) -> dict:
    """Normalise a MoCo-v3 / SimMIM state_dict onto a plain timm ViT."""
    out = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        if nk.startswith(("momentum_encoder.", "predictor.", "decoder.")):
            continue
        for pfx in ("base_encoder.", "encoder."):
            if nk.startswith(pfx):
                nk = nk[len(pfx):]
                break
        out[nk] = v
    return out


def _load_ssl_vit(name: str) -> nn.Module:
    """timm ViT trunk + a MoCo-v3 / SimMIM self-supervised checkpoint, head-less."""
    if not TIMM_AVAILABLE:
        raise ImportError(f"'{name}' needs timm.  pip install timm")
    path = _ssl_vit_checkpoint_path(name)
    if path is None:
        searched = ", ".join(str(d) for d in _ssl_vit_search_dirs()) or "(none exist)"
        raise FileNotFoundError(
            f"self-supervised ViT '{name}' needs a local checkpoint - timm has no "
            f"pretrained tag for it.\n    Searched: {searched}\n"
            f"    Download it (see SSL_VIT_LOCAL in guide/hub/models.py for URLs) "
            f"and drop it in checkpoints/vit/{name}.pth[.tar], or set GUIDE_VIT_CKPT."
        )
    model = timm.create_model(SSL_VIT_LOCAL[name], pretrained=False, num_classes=0)
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    state = _remap_ssl_vit_state(state)
    for k in ("pos_embed", "cls_token"):
        if k in state and k in model.state_dict() and state[k].shape != model.state_dict()[k].shape:
            del state[k]
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [m for m in missing if not m.startswith(("head.", "fc_norm."))]
    if len(real_missing) > 20:
        raise RuntimeError(
            f"checkpoint {path} does not look like a {SSL_VIT_LOCAL[name]} trunk "
            f"({len(real_missing)} missing keys, e.g. {real_missing[:5]})"
        )
    return model.eval()


ALL_MODELS = sorted(set(CNN_BUILDERS) | set(VIT_BUILDERS) | set(SSL_CNN_MODELS))


def load_model(model_name: str) -> nn.Module:
    name = model_name.lower()
    if name in SSL_CNN_MODELS:
        return _load_ssl_cnn(name)
    if name in ITM_EFFICIENTNET:
        return _load_efficientnet_pytorch(ITM_EFFICIENTNET[name])
    if name in CNN_SPECS:
        builder, _enum, extra = CNN_SPECS[name]
        model = builder(weights=_weights_enum(name, config.WEIGHT_VERSION), **extra)
        if not config.KEEP_AUX_HEADS and name in ("googlenet", "inception_v3"):
            model.aux_logits = False
            if hasattr(model, "AuxLogits"):
                model.AuxLogits = None
            if hasattr(model, "aux1"):
                model.aux1 = model.aux2 = None
        return model.eval()

    if name in SSL_VIT_LOCAL:
        return _load_ssl_vit(name)

    if name in VIT_BUILDERS:
        if not TIMM_AVAILABLE:
            raise ImportError(
                f"'{model_name}' needs timm.  pip install timm"
            )
        kwargs = {"pretrained": True}
        if name in SSL_VIT_KEYS:
            kwargs["num_classes"] = 0
        last_error: Exception | None = None
        for candidate in VIT_BUILDERS[name]:
            try:
                return timm.create_model(candidate, **kwargs).eval()
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"could not load '{model_name}' from timm: {last_error}")

    raise ValueError(
        f"unknown model {model_name!r}. known: {', '.join(ALL_MODELS)}"
    )


def image_size_for(model_name: str) -> int:
    return IMAGE_SIZE_OVERRIDES.get(model_name.lower(), config.IMAGE_SIZE)


def source_top1(model_name: str, version: str = config.WEIGHT_VERSION) -> Optional[float]:
    """Published top-1 of the checkpoint actually loaded (V1 unless overridden)."""
    name = model_name.lower()
    if str(version).lower() in ("default", "v2", "best") and name in SOURCE_TOP1_V2:
        return SOURCE_TOP1_V2[name]
    return SOURCE_TOP1.get(name)


def is_headless(model_name: str) -> bool:
    return model_name.lower() in SSL_VIT_KEYS or model_name.lower() in SSL_CNN_MODELS


def find_module_name(model: nn.Module, target: nn.Module) -> Optional[str]:
    for name, module in model.named_modules():
        if module is target:
            return name
    return None


def resolve_classifier(model: nn.Module) -> Optional[nn.Linear]:
    """The final `nn.Linear` head across torchvision CNNs and timm ViT/Swin/PVT."""
    heads = getattr(model, "heads", None)
    if heads is not None and isinstance(getattr(heads, "head", None), nn.Linear):
        return heads.head
    head = getattr(model, "head", None)
    if isinstance(head, nn.Linear):
        return head
    if head is not None and isinstance(getattr(head, "fc", None), nn.Linear):
        return head.fc
    if isinstance(getattr(model, "fc", None), nn.Linear):
        return model.fc
    if isinstance(getattr(model, "_fc", None), nn.Linear):
        return model._fc
    classifier = getattr(model, "classifier", None)
    if isinstance(classifier, nn.Linear):
        return classifier
    if isinstance(classifier, nn.Sequential) and len(classifier):
        for m in reversed(classifier):
            if isinstance(m, nn.Linear):
                return m
    return None


def find_last_mlp_fc2(model: nn.Module) -> Optional[nn.Module]:
    """Penultimate weight matrix for head-less transformers (last block's fc2)."""
    candidates = [
        m
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear)
        and "mlp" in n.lower()
        and ("fc2" in n.lower() or n.lower().endswith(".3"))
    ]
    return candidates[-1] if candidates else None


def find_last_weighted_module(model: nn.Module, exclude: set[str]):
    """Last Linear/Conv module that is not the head."""
    last = (None, None)
    for n, m in model.named_modules():
        if n in exclude:
            continue
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            last = (m, n)
    return last


def weight_matrix(module: nn.Module):
    """2-D view of a module's weight (conv kernels flattened to [out, -1])."""
    w = getattr(module, "weight", None)
    if not isinstance(w, torch.Tensor):
        return None
    if isinstance(module, nn.Linear):
        return w.detach().float().cpu().numpy()
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return w.detach().float().reshape(w.shape[0], -1).cpu().numpy()
    return None


def extract_weight_matrices(model: nn.Module, model_name: str) -> dict:
    """`{"W_classifier": [C, D], "W_penultimate": [.., ..]}` (either may be absent)."""
    out: dict = {}
    clf = resolve_classifier(model)
    exclude: set[str] = set()
    if clf is not None:
        name = find_module_name(model, clf)
        if name:
            exclude.add(name)
        w = weight_matrix(clf)
        if w is not None:
            out["W_classifier"] = w

    pen_mod = None
    if is_headless(model_name) or clf is None:
        pen_mod = find_last_mlp_fc2(model)
    if pen_mod is None:
        pen_mod, _ = find_last_weighted_module(model, exclude)
    if pen_mod is not None:
        w = weight_matrix(pen_mod)
        if w is not None:
            out["W_penultimate"] = w
    return out


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_flops(model: nn.Module, image_size: int) -> Optional[float]:
    """Forward FLOPs, if fvcore or thop happens to be installed."""
    x = torch.zeros(1, 3, image_size, image_size)
    try:
        from fvcore.nn import FlopCountAnalysis

        return float(FlopCountAnalysis(model, x).unsupported_ops_warnings(False).total())
    except Exception:
        pass
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs) * 2.0
    except Exception:
        return None
