"""Text error metrics for ASR evaluation."""

from collections.abc import Sequence


def edit_distance(reference: str, hypothesis: str) -> int:
    """Levenshtein distance using O(min(n, m)) memory."""
    return _sequence_edit_distance(list(reference), list(hypothesis))


def _sequence_edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference
    previous = list(range(len(hypothesis) + 1))
    for row, ref_item in enumerate(reference, start=1):
        current = [row]
        for column, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def error_rates(references: list[str], hypotheses: list[str]) -> dict[str, float]:
    """Return corpus-level character error rate (CER) and word error rate (WER)."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have equal length")
    character_errors = sum(edit_distance(ref, hyp) for ref, hyp in zip(references, hypotheses))
    character_count = sum(len(ref) for ref in references)
    word_refs = [ref.split() for ref in references]
    word_hyps = [hyp.split() for hyp in hypotheses]
    word_errors = sum(_sequence_edit_distance(ref, hyp) for ref, hyp in zip(word_refs, word_hyps))
    word_count = sum(len(ref) for ref in word_refs)
    return {
        "cer": character_errors / character_count if character_count else 0.0,
        "wer": word_errors / word_count if word_count else 0.0,
    }
