import json, os
ckpt_dir = "cache/video_checkpoints"
for d in sorted(os.listdir(ckpt_dir)):
    full = os.path.join(ckpt_dir, d)
    if not os.path.isdir(full):
        continue
    meta = json.load(open(os.path.join(full, "job.json")))
    n_lat = len([f for f in os.listdir(full) if f.startswith("latents")])
    print(f"{d}: {n_lat}/{meta.get('steps','?')} steps, status={meta.get('status','?')}")
