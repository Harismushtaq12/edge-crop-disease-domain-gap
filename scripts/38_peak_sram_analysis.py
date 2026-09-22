"""
Peak activation working set ("SRAM ceiling") for each architecture, derived
analytically from the network graph.

Why this exists. The paper's honest headline limitation is that no physical
embedded device was available, so latency is a CPU proxy. But the number
that actually decides whether a model can run on a Cortex-M class part is
not latency -- it is the peak RAM the activation arena needs, because
weights live in flash and can be streamed, while the activation arena
cannot. That quantity is a property of the *graph*, not of any particular
board, so it can be computed exactly here with no hardware at all.

Metric. We report the standard TinyML per-operator working set used by the
MCUNet line of work:

    peak = max over operators of ( sum(input tensor bytes) + output bytes )

i.e. the largest amount of activation memory that must be simultaneously
resident to execute any single operator. For INT8 deployment one element is
one byte. This is a *lower bound* on a real deployment's RAM: a real
runtime additionally holds long-lived residual/skip tensors across a block,
plus per-kernel scratch, im2col buffers and alignment padding. It is
reported as a bound, and it does not replace on-device validation -- it
makes the deployment claim checkable instead of hand-waved.

Implementation note: the from-scratch checkpoints wrap their backbone as a
nested Functional model, so the layer list must be flattened recursively.
An earlier version of this script walked only the top-level layers and
silently reported the size of the final feature vector instead of the real
arena; the flattening below is what fixes that, and the assertion on the
stem tensor guards against it regressing.

Usage: python3 38_peak_sram_analysis.py
"""
import json
import tensorflow as tf

BYTES_INT8 = 1

MODELS = [
    ("model_tomato_mobilenetv2_fp32.keras",      "MobileNetV2 (α=0.75)",            96),
    ("model_tomato_mobilenetv3small_fp32.keras", "MobileNetV3-Small",               96),
    ("model_tomato_efficientnetb0_fp32.keras",   "EfficientNetB0",                  96),
    ("model_tomato_pretrained_fp32.keras",       "MobileNetV2 (α=1.0, pretrained)", 96),
]

# Reference SRAM of real TinyML-class parts, so the paper can say plainly
# which of these models fits and which does not.
BUDGETS = {
    "Arduino Nano 33 BLE (nRF52840)": 256,
    "Raspberry Pi Pico (RP2040)": 264,
    "STM32F746 Discovery": 320,
    "STM32H743": 1024,
}


def flatten_layers(model):
    """Recursively expand nested Functional/Sequential sub-models."""
    out = []
    for l in model.layers:
        if isinstance(l, tf.keras.Model) and getattr(l, "layers", None):
            out.extend(flatten_layers(l))
        else:
            out.append(l)
    return out


def nbytes(t, itemsize=BYTES_INT8):
    shape = tuple(t.shape)
    n = 1
    for d in shape[1:]:          # drop batch
        if d is None:
            return 0
        n *= int(d)
    return n * itemsize


def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, (list, tuple)) else [x]


def peak_working_set(model):
    """max over ops of (sum of input tensor bytes + output tensor bytes)."""
    peak, where = 0, None
    for l in flatten_layers(model):
        try:
            ins = as_list(l.input)
            outs = as_list(l.output)
        except Exception:
            continue
        b = sum(nbytes(t) for t in ins) + sum(nbytes(t) for t in outs)
        if b > peak:
            peak, where = b, l.name
    return peak, where


rows = []
for path, label, img in MODELS:
    model = tf.keras.models.load_model(path)
    flat = flatten_layers(model)
    peak, where = peak_working_set(model)

    # guard against the nested-model bug: a 96x96x3 input alone is 27 KB, so
    # any real arena for these networks must comfortably exceed that.
    assert peak > 27 * 1024, f"{label}: implausible peak {peak}B - graph not flattened?"

    params = int(model.count_params())
    rec = {
        "model": path, "label": label, "img_size": img,
        "n_layers_flat": len(flat),
        "params": params,
        "weights_int8_kb": round(params * BYTES_INT8 / 1024.0, 1),
        "peak_activation_int8_kb": round(peak / 1024.0, 1),
        "peak_at_layer": where,
        "fits": {k: bool(peak / 1024.0 <= v) for k, v in BUDGETS.items()},
    }
    rows.append(rec)
    print(f"{label:34s} layers={len(flat):3d} params={params/1e6:.2f}M  "
          f"weights(INT8)={rec['weights_int8_kb']:7.1f} KB  "
          f"peak arena={rec['peak_activation_int8_kb']:7.1f} KB  @{where}")
    del model
    tf.keras.backend.clear_session()

print("\nFits within SRAM budget (activations only, weights in flash):")
hdr = f"{'model':34s}" + "".join(f"{k.split(' (')[0][:14]:>16s}" for k in BUDGETS)
print(hdr)
for r in rows:
    print(f"{r['label']:34s}" + "".join(f"{('yes' if r['fits'][k] else 'NO'):>16s}" for k in BUDGETS))

with open("peak_sram_results.json", "w") as f:
    json.dump({
        "metric": "max over operators of (sum input bytes + output bytes), INT8",
        "note": "analytical lower bound: excludes long-lived residual tensors, "
                "per-kernel scratch, im2col buffers and alignment padding",
        "reference_budgets_kb": BUDGETS,
        "variants": rows,
    }, f, indent=2)
print("DONE -> peak_sram_results.json")
