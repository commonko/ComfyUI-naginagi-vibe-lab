"""Generate bundled HumanGate example workflows."""
from __future__ import annotations

import json
import os

from _workflow_builder import IN, OUT, WF


def build_pause_resume():
    wf = WF()
    load = wf.add("LoadImage", (40, 80), (320, 300), title="0. Load Image", widgets=["input.png", "image"], outputs=[OUT("IMAGE", "IMAGE", 0), OUT("MASK", "MASK", 1)])
    pause = wf.add("HumanGatePauseImage", (420, 80), (360, 180), title="1. Pause / Resume / Stop", widgets=["Paused. Resume or Stop?", 0, True], inputs=[IN("image", "IMAGE")], outputs=[OUT("image", "IMAGE", 0), OUT("decision", "STRING", 1), OUT("gate_id", "STRING", 2)], bgcolor="#26384f")
    prev = wf.add("PreviewImage", (840, 80), (320, 300), title="2. Preview", inputs=[IN("images", "IMAGE")])
    wf.link(load, 0, pause, 0, "IMAGE")
    wf.link(pause, 0, prev, 0, "IMAGE")
    wf.group("Human pause gate", (20, 20, 1160, 400), "#3f789e")
    return wf.to_dict()


def build_image_batch_chooser():
    wf = WF()
    ckpt = wf.add("CheckpointLoaderSimple", (40, 80), (320, 100), title="0. Checkpoint", widgets=["model.safetensors"], outputs=[OUT("MODEL", "MODEL", 0), OUT("CLIP", "CLIP", 1), OUT("VAE", "VAE", 2)])
    pos = wf.add("CLIPTextEncode", (40, 230), (320, 120), title="1. Positive", widgets=["best quality, test image"], inputs=[IN("clip", "CLIP")], outputs=[OUT("CONDITIONING", "CONDITIONING", 0)])
    neg = wf.add("CLIPTextEncode", (40, 390), (320, 120), title="2. Negative", widgets=["low quality"], inputs=[IN("clip", "CLIP")], outputs=[OUT("CONDITIONING", "CONDITIONING", 0)])
    latent = wf.add("EmptyLatentImage", (420, 80), (320, 120), title="3. Empty Latent batch 4", widgets=[512, 512, 4], outputs=[OUT("LATENT", "LATENT", 0)])
    ks = wf.add("KSampler", (420, 250), (360, 320), title="4. KSampler batch", widgets=[123456789, 20, 6.0, "euler", "normal", 1.0], inputs=[IN("model", "MODEL"), IN("positive", "CONDITIONING"), IN("negative", "CONDITIONING"), IN("latent_image", "LATENT")], outputs=[OUT("LATENT", "LATENT", 0)])
    dec = wf.add("VAEDecode", (840, 250), (320, 80), title="5. Decode batch", inputs=[IN("samples", "LATENT"), IN("vae", "VAE")], outputs=[OUT("IMAGE", "IMAGE", 0)])
    chooser = wf.add("HumanGateImageChooser", (1200, 180), (380, 240), title="6. Choose image(s)", widgets=["Select image(s), then Resume.", "single", "always_pause", 0], inputs=[IN("images", "IMAGE")], outputs=[OUT("images", "IMAGE", 0), OUT("selected_indices", "STRING", 1), OUT("selection_json", "STRING", 2)], bgcolor="#26384f")
    prev = wf.add("PreviewImage", (1640, 180), (320, 300), title="7. Preview Selected", inputs=[IN("images", "IMAGE")])
    wf.link(ckpt, 1, pos, 0, "CLIP")
    wf.link(ckpt, 1, neg, 0, "CLIP")
    wf.link(ckpt, 0, ks, 0, "MODEL")
    wf.link(pos, 0, ks, 1, "CONDITIONING")
    wf.link(neg, 0, ks, 2, "CONDITIONING")
    wf.link(latent, 0, ks, 3, "LATENT")
    wf.link(ks, 0, dec, 0, "LATENT")
    wf.link(ckpt, 2, dec, 1, "VAE")
    wf.link(dec, 0, chooser, 0, "IMAGE")
    wf.link(chooser, 0, prev, 0, "IMAGE")
    wf.group("Generate batch", (20, 20, 1120, 580), "#6a4a1d")
    wf.group("Human choose", (1180, 120, 800, 400), "#3f789e")
    return wf.to_dict()


def build_pick_image_input():
    wf = WF()
    img1 = wf.add("LoadImage", (40, 80), (320, 300), title="0. Image A", widgets=["input_a.png", "image"], outputs=[OUT("IMAGE", "IMAGE", 0), OUT("MASK", "MASK", 1)])
    img2 = wf.add("LoadImage", (40, 440), (320, 300), title="1. Image B", widgets=["input_b.png", "image"], outputs=[OUT("IMAGE", "IMAGE", 0), OUT("MASK", "MASK", 1)])
    pick = wf.add("HumanGatePickImage", (460, 220), (380, 280), title="2. Pick Input Image", widgets=["Pick one image input.", "A,B", "always_pause", 0], inputs=[IN("image_1", "IMAGE"), IN("image_2", "IMAGE"), IN("image_3", "IMAGE"), IN("image_4", "IMAGE")], outputs=[OUT("image", "IMAGE", 0), OUT("selected_index", "INT", 1), OUT("selected_label", "STRING", 2), OUT("selection_json", "STRING", 3)], bgcolor="#26384f")
    prev = wf.add("PreviewImage", (900, 220), (320, 300), title="3. Preview Picked", inputs=[IN("images", "IMAGE")])
    wf.link(img1, 0, pick, 0, "IMAGE")
    wf.link(img2, 0, pick, 1, "IMAGE")
    wf.link(pick, 0, prev, 0, "IMAGE")
    wf.group("Pick from inputs", (20, 20, 1220, 760), "#3f789e")
    return wf.to_dict()


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    workflows = {
        "01_pause_resume.json": build_pause_resume(),
        "02_image_batch_chooser.json": build_image_batch_chooser(),
        "03_pick_image_input.json": build_pick_image_input(),
    }
    for name, workflow in workflows.items():
        path = os.path.join(here, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(workflow, f, ensure_ascii=False, indent=2)
        print(f"wrote {path}: {len(workflow['nodes'])} nodes, {len(workflow['links'])} links")


if __name__ == "__main__":
    main()
