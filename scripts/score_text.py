"""Score Korean article text with a fine-tuned LEPOS KoELECTRA classifier.

The research pipeline (see ``notebooks/05_model_fine_tuning``) produced a
cascade of KoELECTRA classifiers: A0 (irrelevant-article filter), A2 (ESG
relevance), A3 (ESG sentiment: negative/neutral/positive), and C (E/S/G
pillar). This script runs inference with any of those checkpoints so the
labeling step can be reproduced outside Colab.

Usage::

    python scripts/score_text.py --model <checkpoint-dir-or-hf-id> \
        --labels 부정,중립,긍정 \
        "삼성전자가 재생에너지 100% 전환 계획을 발표했다."

Requires ``transformers`` and ``torch`` (not part of the app requirements)::

    pip install transformers torch
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify Korean text with a fine-tuned LEPOS KoELECTRA model.")
    parser.add_argument(
        "--model", required=True,
        help="Path to a fine-tuned checkpoint directory or a Hugging Face model id")
    parser.add_argument(
        "--labels", default=None,
        help="Optional comma-separated label names in class-index order "
             "(e.g. '부정,중립,긍정' for the A3 sentiment model)")
    parser.add_argument("texts", nargs="+", help="One or more texts to classify")
    args = parser.parse_args()

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        sys.exit("transformers/torch are required: pip install transformers torch")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model)
    model.eval()

    labels = args.labels.split(",") if args.labels else None

    for text in args.texts:
        inputs = tokenizer(text, return_tensors="pt",
                           truncation=True, max_length=512)
        with torch.no_grad():
            probabilities = model(**inputs).logits.softmax(dim=-1)[0]
        prediction = int(probabilities.argmax())
        name = labels[prediction] if labels and prediction < len(labels) else str(prediction)
        confidence = float(probabilities[prediction])
        print(f"[{name} {confidence:.3f}] {text}")


if __name__ == "__main__":
    main()
