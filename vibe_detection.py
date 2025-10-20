"""
Vibe Detection with KPoEM-style labels

IDE meta (non-functional):
- tool: codex
- model: gpt5
- inference: high
- mcp: exa search

This script performs line-by-line emotion (vibe) inference for Korean lyrics.
It uses a Hugging Face transformers pipeline. By default it runs zero-shot
classification with a multilingual NLI model and KPoEM-style label set.

If you have a model fine-tuned on the KPoEM dataset, set MODEL_NAME to that
model (e.g., via environment variable MODEL_NAME or CLI argument --model).
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import List, Dict, Any

import pandas as pd

try:
    from transformers import pipeline, AutoTokenizer
except Exception as e:
    print("[ERROR] transformers is required. Install with: pip install transformers pandas", file=sys.stderr)
    raise


# KPoEM-style label set (Korean)
LABELS: List[str] = [
    '불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움', '슬픔', '화남/분노', '존경',
    '기대감', '우쭐댐/무시함', '안타까움/실망', '비장함', '의심/불신', '뿌듯함', '편안/쾌적',
    '신기함/관심', '아껴주는', '부끄러움', '공포/무서움', '절망', '한심함', '역겨움/징그러움',
    '짜증', '어이없음', '없음', '패배/자기혐오', '귀찮음', '힘듦/지침', '즐거움/신남', '깨달음',
    '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', '당황/난처', '경악', '부담/안_내킴', '서러움',
    '재미없음', '불쌍함/연민', '놀람', '행복', '불안/걱정', '기쁨', '안심/신뢰'
]


DEFAULT_ZERO_SHOT_MODEL = os.environ.get(
    "MODEL_NAME",
    # Smaller multilingual NLI model to reduce download size.
    # Swap to your fine-tuned KPoEM model if available.
    "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)


def build_classifier(model_name: str) -> Any:
    """Create a transformers pipeline for vibe detection.

    If `model_name` points to a sequence classification model fine-tuned for
    KPoEM labels, you can instead build a text-classification pipeline and
    directly map logits to labels. For broad compatibility without requiring
    a specific fine-tuned model, we use zero-shot classification with the
    provided KPoEM-style labels.
    """
    # Prefer fast tokenizer to avoid slow->fast conversion dependencies
    tok = None
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e_fast:
        # Fallback to slow tokenizer (requires sentencepiece for some models)
        try:
            tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        except Exception as e_slow:
            missing_hints = [
                "pip install -U protobuf",
                "pip install -U sentencepiece",
                "pip install -U tiktoken",
            ]
            msg = (
                "Tokenizer 로드에 실패했습니다. 필요한 패키지가 누락되었을 수 있습니다.\n"
                f"model={model_name}\n"
                f"fast_error={type(e_fast).__name__}: {e_fast}\n"
                f"slow_error={type(e_slow).__name__}: {e_slow}\n"
                "다음 설치를 시도해보세요:\n  - " + "\n  - ".join(missing_hints)
            )
            raise RuntimeError(msg) from e_slow

    clf = pipeline(task="zero-shot-classification", model=model_name, tokenizer=tok)
    return clf


def infer_vibes(
    lines: List[str],
    clf: Any,
    labels: List[str],
    hypothesis_template: str = "이 문장은 {} 감정이다.",
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for idx, text in enumerate(lines, start=1):
        if not text.strip():
            continue
        out = clf(
            text,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )

        # Standardize output
        candidate_labels = out.get("labels", [])
        scores = out.get("scores", [])
        if not candidate_labels or not scores:
            pred_label = ""; pred_score = 0.0
            top_items: List[str] = []
        else:
            pred_label = candidate_labels[0]
            pred_score = float(scores[0])
            k = min(top_k, len(candidate_labels))
            top_items = [f"{candidate_labels[i]}:{scores[i]:.4f}" for i in range(k)]

        print(f"[{idx}] {text} -> {pred_label} ({pred_score:.4f})")

        results.append({
            "line_no": idx,
            "text": text,
            "pred_label": pred_label,
            "pred_score": pred_score,
            "topk": ", ".join(top_items),
        })
    return results


def read_lyrics_from_stdin() -> str:
    print(
        "가사를 붙여넣고, 끝내려면 빈 줄에서 ENTER 두 번 또는 'END' 입력 후 ENTER를 누르세요.",
        file=sys.stderr,
    )
    buf: List[str] = []
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not line:  # EOF
            break
        # Normalize Windows newlines
        line = line.rstrip("\r\n")
        if line.strip() == "END":
            break
        buf.append(line)
        # Stop on double-ENTER (empty line followed by ENTER)
        if len(buf) >= 2 and buf[-1] == "" and buf[-2] == "":
            # Remove the trailing empty line
            while buf and buf[-1] == "":
                buf.pop()
            break
    return "\n".join(buf)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Korean Vibe Detection (KPoEM-style labels)")
    parser.add_argument("--model", type=str, default=DEFAULT_ZERO_SHOT_MODEL,
                        help="Hugging Face model ID (default: zero-shot multilingual NLI model). "
                             "If you have a KPoEM-finetuned model, set it here.")
    parser.add_argument("--input", type=str, default=None,
                        help="Optional path to a lyrics text file. If omitted, reads from stdin.")
    parser.add_argument("--out", type=str, default="vibe_results.csv",
                        help="Output CSV path (CP949-encoded).")
    parser.add_argument("--topk", type=int, default=3, help="Top-K labels to include in CSV.")
    parser.add_argument("--template", type=str, default="이 문장은 {} 감정이다.",
                        help="Zero-shot hypothesis template.")
    args = parser.parse_args(argv)

    # Read lyrics
    if args.input:
        # Try UTF-8 first, then fall back to CP949 for Windows-origin files
        try:
            with open(args.input, "r", encoding="utf-8") as f:
                lyrics = f.read()
        except UnicodeDecodeError:
            with open(args.input, "r", encoding="cp949", errors="replace") as f:
                lyrics = f.read()
    else:
        lyrics = read_lyrics_from_stdin()

    lines = lyrics.split("\n")
    # Build classifier
    clf = build_classifier(args.model)

    print(f"\n[INFO] 모델: {args.model}")
    print(f"[INFO] 줄별 감정(Emotion/Vibe) 예측을 시작합니다. 총 {len([l for l in lines if l.strip()])}개 라인")

    rows = infer_vibes(
        lines=lines,
        clf=clf,
        labels=LABELS,
        hypothesis_template=args.template,
        top_k=args.topk,
    )

    if not rows:
        print("[WARN] 유효한 라인이 없어 CSV를 생성하지 않습니다.")
        return 0

    df = pd.DataFrame(rows, columns=["line_no", "text", "pred_label", "pred_score", "topk"])
    # Ensure CP949 for Korean compatibility on Windows. Fallback to UTF-8-SIG on encoding errors.
    try:
        df.to_csv(args.out, index=False, encoding="cp949")
        print(f"\n[OK] 결과 저장 완료: {args.out} (encoding=cp949)")
    except UnicodeEncodeError:
        alt_out = os.path.splitext(args.out)[0] + "_utf8.csv"
        df.to_csv(alt_out, index=False, encoding="utf-8-sig")
        print(f"\n[OK] 결과 저장 완료: {alt_out} (encoding=utf-8-sig; CP949 변환 불가 문자 포함)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
