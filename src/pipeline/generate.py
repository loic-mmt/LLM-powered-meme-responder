import json
import re
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Ensure project root is on PYTHONPATH when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm.client import OllamaClient
from src.render import draw_bottom_caption

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOPK = 8


def normalize(t: str) -> str:
    t = t.lower().strip()
    t = re.sub(r"[^\w\s']", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t


def load_memes():
    memes = []
    with open("memes/metadata.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                # If metadata says .jpg but your assets are .jpeg
                if "file" in m and m["file"].lower().endswith(".jpg"):
                    m["file"] = m["file"][:-4] + ".jpeg"
                memes.append(m)
    by_id = {m["id"]: m for m in memes}
    return memes, by_id


def rule_match(prompt_norm: str, memes):
    for m in memes:
        for trig in m.get("triggers", []):
            if normalize(trig) in prompt_norm:
                return m["id"]
    return None


def topk_by_embeddings(prompt_norm: str, k: int = TOPK):
    emb = np.load("indexes/meme_emb.npy")
    ids = json.loads(Path("indexes/meme_ids.json").read_text(encoding="utf-8"))

    model = SentenceTransformer(EMB_MODEL)
    q = model.encode([prompt_norm], normalize_embeddings=True)[0]
    scores = emb @ q
    idxs = np.argsort(-scores)[:k]
    return [(ids[i], float(scores[i])) for i in idxs]


def caption_with_llm(client: OllamaClient, user_prompt: str, meme: dict) -> str:
    tags = meme.get("tags", [])
    prompt = f"""
Return ONLY JSON: {{\"caption\":\"...\"}}.

Rules:
- English unless the user prompt is clearly French.
- Max 80 characters.
- No emojis.
- Match the vibe tags: {tags}
- Don't integrate the tags inside the caption.
- Make it meme-like: short, punchy.

USER_PROMPT: {user_prompt}
"""
    raw = client.complete(prompt, temperature=0.2, max_tokens=160).strip()

    try:
        s = raw[raw.find("{") : raw.rfind("}") + 1]
        obj = json.loads(s)
        cap = (obj.get("caption") or "").strip()
        if not cap:
            return "..."
        return cap[:80]
    except Exception:
        return "..."


def generate(prompt: str, out_path: str = "outputs/out.png", default_id: str = "ok_tlof"):
    memes, by_id = load_memes()
    pnorm = normalize(prompt)

    # 1) triggers
    meme_id = rule_match(pnorm, memes)

    # 2) retrieval
    if not meme_id:
        candidates = topk_by_embeddings(pnorm, k=TOPK)
        meme_id = candidates[0][0] if candidates else default_id

    meme = by_id.get(meme_id) or by_id.get(default_id)
    if meme is None:
        raise RuntimeError("No default meme found. Set default_id to an existing meme id.")

    client = OllamaClient(model="qwen2.5:3b-instruct")
    caption = caption_with_llm(client, prompt, meme)

    img_path = PROJECT_ROOT / "memes" / "images" / meme["file"]
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    text_cfg = meme.get("text", {})
    draw_bottom_caption(
        str(img_path),
        caption,
        out_path,
        font_size=text_cfg.get("font_size", 32),
        stroke=text_cfg.get("stroke", 4),
    )

    return {"meme_id": meme["id"], "caption": caption, "file": out_path}


if __name__ == "__main__":
    user_prompt = " ".join(sys.argv[1:]).strip() or input("Prompt: ").strip()
    res = generate(user_prompt, out_path="outputs/out.png")
    print("Done.", res)