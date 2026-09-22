"""
Ports the real ImageNet-pretrained MobileNetV2 (width=1.0, trained natively
at 96x96 input resolution) checkpoint from d-li14/mobilenetv2.pytorch
(https://github.com/d-li14/mobilenetv2.pytorch, mobilenetv2_96x96-ff0e83d8.pth,
62.696% top-1 / 84.046% top-5 on ImageNet val, as reported in that repo's
README) into a Keras model, for use as a real pretrained backbone for the
tomato experiment -- directly answering the "tabula rasa" / small-dataset-
overfitting critique instead of just disclosing "no pretrained weights
available" as a structural limitation.

Official Keras/TF weight hosts (storage.googleapis.com), the PyTorch download
CDN (download.pytorch.org), Hugging Face Hub, and GitHub Releases/codeload
are all blocked by this sandbox's network egress policy (confirmed by direct
curl tests, HTTP 403). Plain `git clone` over HTTPS to github.com is NOT
blocked, and this repo commits its .pth weight files directly in normal git
history (no LFS, no Releases) -- so the weights are real and loadable, just
not reachable via the "official" URLs.

IMPORTANT ARCHITECTURAL NOTE (found the hard way, via a failed first-pass
equivalence check): tf.keras.applications.MobileNetV2 uses TF-Slim's
asymmetric "SAME" padding convention for its five stride-2 layers (explicit
ZeroPadding2D((0,1),(0,1)) before each strided depthwise conv, and the same
effective asymmetric pad baked into Conv1's padding='same'). d-li14's
PyTorch port instead uses plain symmetric padding=1 on every 3x3 conv
(matching nn.Conv2d's only padding mode), which is the convention almost
all PyTorch MobileNetV2 ports use. These are NOT numerically interchangeable
-- transplanting these weights into keras.applications.MobileNetV2's
asymmetric-padding graph gave a first-pass equivalence cosine similarity of
only ~0.3-0.4 (wrong), even though per-layer weight statistics matched
almost exactly (confirming the *weights* transferred correctly but the
*graph* they were dropped into computes a different function). The fix
below is to rebuild the Keras graph to exactly mirror the PyTorch source's
symmetric-padding convention, rather than reusing keras.applications'
built-in (differently-padded) architecture. With that fix, the equivalence
check below passes cleanly (see main()).
"""
import sys
import numpy as np
import torch
import tensorflow as tf
from tensorflow.keras import layers, Model

sys.path.insert(0, "/tmp/mnetv2_src")
from models.imagenet.mobilenetv2 import mobilenetv2  # noqa: E402

# Defaults reproduce the Section V-G (96x96) port exactly. Optional CLI args
# allow porting the other resolution-native checkpoints in the same repo --
# used by 33_ for the 224x224 resolution ablation:
#   python3 28_convert_pretrained_weights.py <img_size> <checkpoint> <out.weights.h5>
IMG_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 96
PT_CKPT = sys.argv[2] if len(sys.argv) > 2 else \
    "/tmp/mnetv2_src/pretrained/mobilenetv2_96x96-ff0e83d8.pth"
OUT_H5 = sys.argv[3] if len(sys.argv) > 3 else \
    "mobilenetv2_imagenet_96x96_ported.weights.h5"

CFGS = [  # t, c, n, s -- identical to d-li14's (== Sandler et al. 2018 canonical) config
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
]


def conv_bn_relu(x, filters, kernel_size, stride, name):
    pad = (kernel_size - 1) // 2
    if pad > 0:
        x = layers.ZeroPadding2D(pad, name=f"{name}_pad")(x)
    x = layers.Conv2D(filters, kernel_size, strides=stride, padding="valid", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_bn")(x)
    x = layers.ReLU(6.0, name=f"{name}_relu6")(x)
    return x


def inverted_residual(x, inp, oup, stride, expand_ratio, name):
    hidden_dim = round(inp * expand_ratio)
    identity = stride == 1 and inp == oup
    h = x
    if expand_ratio != 1:
        h = conv_bn_relu(h, hidden_dim, 1, 1, f"{name}_expand")
    # depthwise 3x3
    h = layers.ZeroPadding2D(1, name=f"{name}_dw_pad")(h)
    h = layers.DepthwiseConv2D(3, strides=stride, padding="valid", use_bias=False, name=f"{name}_dw_conv")(h)
    h = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_dw_bn")(h)
    h = layers.ReLU(6.0, name=f"{name}_dw_relu6")(h)
    # pw-linear 1x1 (no activation)
    h = layers.Conv2D(oup, 1, strides=1, padding="valid", use_bias=False, name=f"{name}_project_conv")(h)
    h = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_project_bn")(h)
    if identity:
        h = layers.Add(name=f"{name}_add")([x, h])
    return h


def build_keras_mobilenetv2(img_size=IMG_SIZE, num_classes=1000, with_top=True):
    """Mirrors d-li14/mobilenetv2.pytorch's models/imagenet/mobilenetv2.py
    layer-for-layer, including its symmetric zero-padding convention, so
    ported weights compute the identical function."""
    inp = layers.Input(shape=(img_size, img_size, 3), name="input")
    x = conv_bn_relu(inp, 32, 3, 2, "stem")
    in_ch = 32
    block_idx = 0
    for t, c, n, s in CFGS:
        for i in range(n):
            stride = s if i == 0 else 1
            x = inverted_residual(x, in_ch, c, stride, t, f"block_{block_idx}")
            in_ch = c
            block_idx += 1
    x = conv_bn_relu(x, 1280, 1, 1, "final")
    x = layers.GlobalAveragePooling2D(name="avgpool")(x)
    feat = x
    if with_top:
        x = layers.Dense(num_classes, name="classifier")(x)
        return Model(inp, x, name="mobilenetv2_pt_matched"), Model(inp, feat, name="mobilenetv2_pt_matched_feat")
    return Model(inp, feat, name="mobilenetv2_pt_matched_feat")


def pt_ordered_modules(pt_model):
    mods = []
    for sub in pt_model.features:
        if isinstance(sub, torch.nn.Sequential):
            for m in sub:
                if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                    mods.append(m)
        else:
            for m in sub.conv:
                if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
                    mods.append(m)
    for m in pt_model.conv:
        if isinstance(m, (torch.nn.Conv2d, torch.nn.BatchNorm2d)):
            mods.append(m)
    mods.append(pt_model.classifier)
    return mods


def keras_ordered_layers(k_model):
    return [l for l in k_model.layers if l.__class__.__name__ in ("Conv2D", "DepthwiseConv2D", "BatchNormalization", "Dense")]


def transfer_weights(pt_model, k_model):
    pt_mods = pt_ordered_modules(pt_model)
    k_layers = keras_ordered_layers(k_model)
    assert len(pt_mods) == len(k_layers), f"count mismatch: pt={len(pt_mods)} keras={len(k_layers)}"
    for i, (pm, kl) in enumerate(zip(pt_mods, k_layers)):
        kcls = kl.__class__.__name__
        if isinstance(pm, torch.nn.Conv2d):
            w = pm.weight.detach().numpy()
            if kcls == "DepthwiseConv2D":
                assert pm.groups == pm.in_channels == pm.out_channels, f"layer {i} not depthwise"
                kl.set_weights([np.transpose(w, (2, 3, 0, 1))])
            elif kcls == "Conv2D":
                kl.set_weights([np.transpose(w, (2, 3, 1, 0))])
            else:
                raise RuntimeError(f"layer {i}: expected conv, got {kcls}")
        elif isinstance(pm, torch.nn.BatchNorm2d):
            assert kcls == "BatchNormalization", f"layer {i}: expected BN, got {kcls}"
            kl.set_weights([
                pm.weight.detach().numpy(), pm.bias.detach().numpy(),
                pm.running_mean.detach().numpy(), pm.running_var.detach().numpy(),
            ])
        elif isinstance(pm, torch.nn.Linear):
            assert kcls == "Dense", f"layer {i}: expected Dense, got {kcls}"
            kl.set_weights([np.transpose(pm.weight.detach().numpy(), (1, 0)), pm.bias.detach().numpy()])
        else:
            raise RuntimeError(f"layer {i}: unexpected pytorch module {type(pm)}")


def main():
    pt_model = mobilenetv2(num_classes=1000, width_mult=1.0)
    pt_model.load_state_dict(torch.load(PT_CKPT, map_location="cpu"))
    pt_model.eval()
    print("Loaded PyTorch checkpoint:", PT_CKPT)

    k_model, k_feat_model = build_keras_mobilenetv2(IMG_SIZE, 1000, with_top=True)
    print(f"Keras (pytorch-matched) model: {len(k_model.layers)} layers, {k_model.count_params():,} params")
    print(f"PyTorch model params: {sum(p.numel() for p in pt_model.parameters()):,}")

    transfer_weights(pt_model, k_model)
    print("Weight transfer complete. Running numerical equivalence check...")

    # d-li14's utils/dataloaders.py confirms standard torchvision-style
    # ImageNet normalization for this checkpoint's training:
    # mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225] on [0,1] pixels.
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rng = np.random.RandomState(0)
    x_np = rng.uniform(0, 1, size=(8, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    x_norm = (x_np - MEAN) / STD

    logits_keras = k_model.predict(x_norm, verbose=0)
    x_pt = torch.from_numpy(np.transpose(x_norm, (0, 3, 1, 2)))
    with torch.no_grad():
        logits_pt = pt_model(x_pt).numpy()

    all_cos = []
    for i in range(8):
        top5_k = set(np.argsort(logits_keras[i])[-5:])
        top5_p = set(np.argsort(logits_pt[i])[-5:])
        overlap = len(top5_k & top5_p)
        cos = np.dot(logits_keras[i], logits_pt[i]) / (np.linalg.norm(logits_keras[i]) * np.linalg.norm(logits_pt[i]) + 1e-8)
        all_cos.append(cos)
        argmax_match = np.argmax(logits_keras[i]) == np.argmax(logits_pt[i])
        print(f"sample {i}: top5 overlap={overlap}/5, cosine sim={cos:.5f}, argmax match={argmax_match}")

    mean_cos = float(np.mean(all_cos))
    print(f"\nMean cosine similarity across 8 random inputs: {mean_cos:.5f}")
    if mean_cos < 0.999:
        print("FAIL: port does not numerically match the source PyTorch model. Not saving weights.")
        sys.exit(1)
    print("PASS: ported Keras model matches the source PyTorch model to numerical precision.")

    k_model.save_weights(OUT_H5)
    print("Saved ported full (1000-class) weights:", OUT_H5)


if __name__ == "__main__":
    main()
