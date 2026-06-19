from __future__ import annotations

from typing import Any

from src.matching import tokenize


def longest_common_contiguous_ngram(a: list[str], b: list[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for tok_a in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if tok_a == tok_b:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def exact_span_recall(generated: list[str], truth: list[str], span: int = 5) -> float:
    if len(truth) < span:
        return 0.0
    gen_spans = {tuple(generated[i : i + span]) for i in range(0, max(0, len(generated) - span + 1))}
    if not gen_spans:
        return 0.0
    truth_spans = [tuple(truth[i : i + span]) for i in range(0, len(truth) - span + 1)]
    hits = sum(1 for s in truth_spans if s in gen_spans)
    return hits / max(1, len(truth_spans))


def try_local_generator(model_name: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.eval()

        def generate(prompt: str, max_new_tokens: int) -> str:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            continuation = output[0][inputs["input_ids"].shape[1] :]
            return tokenizer.decode(continuation, skip_special_tokens=True)

        return generate, None
    except Exception as exc:  # pragma: no cover - environment fallback
        return None, f"{type(exc).__name__}: {exc}"


def lexical_fallback_generation(prefix_tokens: list[str], max_new_tokens: int) -> str:
    # This is intentionally weak and is marked as a P0 deviation by the report.
    return " ".join(prefix_tokens[-max_new_tokens:])


def contamination_report(
    corpus: list[dict[str, Any]],
    transcript_ids: list[str],
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config["contamination"]
    rows_by_id = {row["transcript_id"]: row for row in corpus}
    generator, generator_error = try_local_generator(cfg["local_model"])
    method = "local_deterministic_generation" if generator else "lexical_fallback_p0_deviation"
    flagged = []
    rows = []
    for tid in transcript_ids[: int(cfg["max_transcripts_for_generation"])]:
        row = rows_by_id[tid]
        toks = tokenize(row["text"])
        prefix_n = min(int(cfg["prefix_words"]), max(1, len(toks) // 3))
        cont_n = int(cfg["continuation_words"])
        if len(toks) <= prefix_n + 20:
            continue
        prefix_tokens = toks[:prefix_n]
        truth_tokens = toks[prefix_n : prefix_n + cont_n]
        prompt = " ".join(prefix_tokens)
        if generator:
            gen_text = generator(prompt, int(cfg["max_new_tokens"]))
        else:
            gen_text = lexical_fallback_generation(prefix_tokens, int(cfg["max_new_tokens"]))
        gen_tokens = tokenize(gen_text)
        lcn = longest_common_contiguous_ngram(gen_tokens, truth_tokens)
        recall = exact_span_recall(gen_tokens, truth_tokens, span=5)
        is_flagged = (
            lcn >= int(cfg["longest_ngram_flag_threshold"])
            or recall >= float(cfg["exact_span_recall_flag_threshold"])
        )
        item = {
            "transcript_id": tid,
            "title": row["title"],
            "date": row["date"],
            "source_url": row["source_url"],
            "longest_common_ngram": int(lcn),
            "exact_span_recall_5gram": float(recall),
            "flagged": bool(is_flagged),
        }
        rows.append(item)
        if is_flagged:
            flagged.append(item)
    return {
        "method": method,
        "model": cfg["local_model"] if generator else None,
        "generator_error": generator_error,
        "prefix_words": int(cfg["prefix_words"]),
        "continuation_words": int(cfg["continuation_words"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "longest_ngram_flag_threshold": int(cfg["longest_ngram_flag_threshold"]),
        "exact_span_recall_flag_threshold": float(cfg["exact_span_recall_flag_threshold"]),
        "n_checked": len(rows),
        "n_flagged": len(flagged),
        "flagged_transcript_ids": [x["transcript_id"] for x in flagged],
        "rows": rows,
        "p0_deviation": method != "local_deterministic_generation",
    }
