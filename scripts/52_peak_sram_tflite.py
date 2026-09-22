"""
Peak activation memory, computed on the actual INT8 TFLite graph.

This replaces 38_peak_sram_analysis.py, which walked the Keras layer list.
That was wrong in a way that mattered. Keras keeps shape-only layers --
ZeroPadding2D, Activation, Rescaling, Reshape -- as separate layers, and the
converter removes or fuses every one of them: padding becomes the
convolution's own padding, activations fuse into the preceding kernel. The
old walk therefore scored a padding "operator" whose input and output are
the same tensor at two spatial sizes, counted both, and reported the sum as
a memory peak. That is why the peak landed on `block_1_pad`,
`block2a_dwconv_pad` and `expanded_conv_1_depthwise_pad` in every model, and
why two quite different graphs, MobileNetV2 and EfficientNetB0, came out at
an identical 441.1 KB: the number was a property of the padding convention,
not of the network.

Here we read the deployed graph instead. The TFLite interpreter exposes
every tensor's shape and dtype and every operator's input/output indices
after all fusions, so what we measure is what would actually be executed.

Two numbers per model:

  peak_op     max over operators of (sum of input bytes + sum of output
              bytes) -- the MCUNet-style per-operator working set: the
              memory that must be simultaneously addressable to run the
              single heaviest kernel.

  peak_arena  max over execution steps of the total bytes of tensors that
              are live at that step, where a tensor is live from the
              operator that writes it until its last reader. This is the
              arena an ideal allocator would need, and it is the number that
              decides whether a model fits in a microcontroller's SRAM. It
              dominates peak_op, so it is the one we report.

Both exclude per-kernel scratch, im2col buffers and alignment padding, so
both remain lower bounds on a real deployment -- but they are lower bounds
on the right graph. Weights are not counted: they live in flash.
"""
import json
import numpy as np
import tensorflow as tf

MODELS = [
    ("model_tomato_mobilenetv2_int8.tflite",      "MobileNetV2 ($\\alpha$=0.75)", 96),
    ("model_tomato_mobilenetv3small_int8.tflite", "MobileNetV3-Small",            96),
    ("model_tomato_efficientnetb0_int8.tflite",   "EfficientNetB0",               96),
    ("model_tomato_pretrained_int8.tflite",       "MobileNetV2 ($\\alpha$=1.0, pretrained)", 96),
]

BUDGETS = {
    "Arduino Nano 33 BLE (nRF52840)": 256,
    "Raspberry Pi Pico (RP2040)": 264,
    "STM32F746 Discovery": 320,
    "STM32H743": 1024,
}


def tensor_bytes(det):
    shape = det["shape"]
    if len(shape) == 0:
        return 0
    n = int(np.prod([int(d) for d in shape]))
    return n * np.dtype(det["dtype"]).itemsize


def analyse(path):
    # The default interpreter lets XNNPACK fuse whole subgraphs into a single
    # DELEGATE node, which hides the per-operator tensor structure we are
    # trying to measure. The reference op resolver keeps every builtin
    # operator separate, which is also what a microcontroller runtime does.
    interp = tf.lite.Interpreter(
        model_path=path,
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    interp.allocate_tensors()
    assert not any(op["op_name"] == "DELEGATE" for op in interp._get_ops_details()), \
        f"{path}: graph is still delegated; per-operator memory is not visible"

    ops = interp._get_ops_details()
    details = {d["index"]: d for d in interp.get_tensor_details()}

    # Constant tensors (weights, biases, quantization params) live in flash,
    # not in the activation arena, so they are excluded. A tensor is constant
    # if no operator writes it.
    produced = set()
    for op in ops:
        produced.update(int(i) for i in op["outputs"] if int(i) >= 0)
    graph_inputs = {int(d["index"]) for d in interp.get_input_details()}
    graph_outputs = {int(d["index"]) for d in interp.get_output_details()}
    activation = produced | graph_inputs

    size = {t: tensor_bytes(details[t]) for t in activation if t in details}

    # last reader of each activation tensor, for the lifetime analysis
    last_use = {}
    for step, op in enumerate(ops):
        for t in op["inputs"]:
            t = int(t)
            if t in activation:
                last_use[t] = step
    for t in graph_outputs:
        last_use[int(t)] = len(ops) - 1

    peak_op, where_op = 0, None
    peak_arena, where_arena = 0, None
    live = set(graph_inputs)
    top = []

    for step, op in enumerate(ops):
        outs = [int(t) for t in op["outputs"] if int(t) in activation]
        ins = [int(t) for t in op["inputs"] if int(t) in activation]

        b_op = sum(size.get(t, 0) for t in ins) + sum(size.get(t, 0) for t in outs)
        if b_op > peak_op:
            peak_op, where_op = b_op, op["op_name"]

        live.update(outs)
        b_arena = sum(size.get(t, 0) for t in live)
        top.append((b_arena, step, op["op_name"]))
        if b_arena > peak_arena:
            peak_arena, where_arena = b_arena, op["op_name"]

        for t in ins:
            if last_use.get(t, -1) <= step and t not in graph_outputs:
                live.discard(t)

    top.sort(reverse=True)
    return {
        "n_ops": len(ops),
        "peak_op_kb": round(peak_op / 1024.0, 1),
        "peak_op_at": where_op,
        "peak_arena_kb": round(peak_arena / 1024.0, 1),
        "peak_arena_at": where_arena,
        "top5_arena": [{"kb": round(b / 1024.0, 1), "step": s, "op": o}
                       for b, s, o in top[:5]],
    }


rows = []
for path, label, img in MODELS:
    r = analyse(path)
    r.update({"model": path, "label": label, "img_size": img})
    r["fits"] = {k: bool(r["peak_arena_kb"] <= v) for k, v in BUDGETS.items()}
    rows.append(r)
    print(f"{label:32s} ops={r['n_ops']:3d}  per-op={r['peak_op_kb']:7.1f} KB "
          f"({r['peak_op_at']})   arena={r['peak_arena_kb']:7.1f} KB "
          f"({r['peak_arena_at']})")
    print("      top arena steps:", ", ".join(
        f"{t['kb']}KB@{t['op']}" for t in r["top5_arena"]))

print("\nFits within SRAM budget (activation arena only; weights in flash):")
for r in rows:
    print(f"{r['label']:32s}" +
          "".join(f"{('yes' if r['fits'][k] else 'NO'):>16s}" for k in BUDGETS))

with open("peak_sram_tflite_results.json", "w") as f:
    json.dump({
        "metric": "peak_arena = max over execution steps of total live "
                  "activation bytes on the converted INT8 TFLite graph",
        "note": "lower bound on deployed RAM: excludes per-kernel scratch, "
                "im2col buffers and allocator alignment padding. Weights "
                "excluded (flash-resident).",
        "reference_budgets_kb": BUDGETS,
        "variants": rows,
    }, f, indent=2)
print("DONE -> peak_sram_tflite_results.json")
