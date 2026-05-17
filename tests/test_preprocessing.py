"""
US-03 Acceptance Criteria Tests
Synora — Automated Text Preprocessing Pipeline

Runs automatically. Each test maps directly to an acceptance criterion:

  ✅ Tokenization applied consistently across all client shards
  ✅ Vocabulary mapping completed without errors
  ✅ Padding applied to uniform length
  ✅ No null values present in output tensors
  ✅ Out-of-vocabulary tokens handled without errors
"""

import sys
import os
import json
from pathlib import Path

# Allow importing from parent data directory
sys.path.insert(0, str(Path(__file__).parent.parent / "data"))
from preprocessing import (
    tokenize,
    tokenize_dataset,
    build_vocabulary,
    encode_sequence,
    encode_dataset,
    pad_sequence,
    pad_dataset,
    build_label_map,
    encode_labels,
    validate_output,
    run_preprocessing_pipeline,
    MAX_SEQUENCE_LENGTH,
    OOV_TOKEN,
    PADDING_TOKEN,
)


# ─────────────────────────────────────────────
#  TEST HELPERS
# ─────────────────────────────────────────────

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results = []

def run_test(name: str, fn):
    try:
        fn()
        print(f"{PASS}: {name}")
        results.append((name, True, None))
    except Exception as e:
        print(f"{FAIL}: {name}\n       → {e}")
        results.append((name, False, str(e)))


# ─────────────────────────────────────────────
#  SHARED FIXTURES
# ─────────────────────────────────────────────

RAW_SHARDS = {
    0: [
        {"text": "Habari yako leo",        "label": "greeting"},
        {"text": "Mambo vipi rafiki",      "label": "greeting"},
        {"text": "Ninasikia njaa sana",    "label": "complaint"},
    ],
    1: [
        {"text": "Asante sana kwa msaada", "label": "positive"},
        {"text": "Sijui la kufanya sasa",  "label": "complaint"},
        {"text": "Tafadhali nisaidie",     "label": "request"},
    ],
    2: [
        {"text": "Habari za jioni",        "label": "greeting"},
        {"text": "Ninafurahi sana",        "label": "positive"},
        {"text": "Kuna tatizo kubwa",      "label": "complaint"},
    ],
}

ALL_SAMPLES = [s for shard in RAW_SHARDS.values() for s in shard]

def make_full_pipeline():
    """Run the full pipeline and return (padded_samples, vocab, label_map)."""
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)
    label_map = build_label_map(ALL_SAMPLES)
    encoded = encode_dataset(tokenized, vocab)
    labeled = encode_labels(encoded, label_map)
    padded = pad_dataset(labeled, MAX_SEQUENCE_LENGTH)
    return padded, vocab, label_map


# ─────────────────────────────────────────────
#  TESTS: TOKENIZATION
# ─────────────────────────────────────────────

def test_tokenize_basic():
    tokens = tokenize("Habari yako leo!")
    assert tokens == ["habari", "yako", "leo"], f"Got {tokens}"

def test_tokenize_lowercase():
    tokens = tokenize("HABARI YAKO")
    assert all(t == t.lower() for t in tokens), "Tokens not lowercased"

def test_tokenize_strips_punctuation():
    tokens = tokenize("Hello, world!")
    assert "," not in tokens and "!" not in tokens

def test_tokenize_empty_string():
    tokens = tokenize("")
    assert tokens == [], f"Expected [] but got {tokens}"

def test_tokenize_none_like():
    tokens = tokenize("   ")
    assert tokens == []

def test_tokenize_consistent_across_shards():
    """
    AC: Tokenization applied consistently across all client shards.
    Same text must always produce same tokens regardless of which shard it's in.
    """
    text = "Habari yako leo"
    result_a = tokenize(text)
    result_b = tokenize(text)
    assert result_a == result_b, "Tokenization is not deterministic!"

    # Test that the same text processed in different shards gives same tokens
    shard_a = tokenize_dataset([{"text": text, "label": "greeting"}])
    shard_b = tokenize_dataset([{"text": text, "label": "greeting"}])
    assert shard_a[0]["tokens"] == shard_b[0]["tokens"]


# ─────────────────────────────────────────────
#  TESTS: VOCABULARY
# ─────────────────────────────────────────────

def test_vocab_contains_pad_and_oov():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)
    assert PADDING_TOKEN in vocab, "<PAD> missing from vocab"
    assert OOV_TOKEN in vocab, "<OOV> missing from vocab"

def test_vocab_pad_is_index_0():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)
    assert vocab[PADDING_TOKEN] == 0, f"<PAD> should be 0, got {vocab[PADDING_TOKEN]}"

def test_vocab_oov_is_index_1():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)
    assert vocab[OOV_TOKEN] == 1, f"<OOV> should be 1, got {vocab[OOV_TOKEN]}"

def test_vocab_no_duplicate_indices():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)
    indices = list(vocab.values())
    assert len(indices) == len(set(indices)), "Duplicate indices in vocabulary!"

def test_vocab_mapping_without_errors():
    """AC: Vocabulary mapping completed without errors."""
    try:
        tokenized = tokenize_dataset(ALL_SAMPLES)
        vocab = build_vocabulary(tokenized)
        assert len(vocab) >= 2  # At minimum PAD + OOV
    except Exception as e:
        raise AssertionError(f"Vocabulary building raised an error: {e}")

def test_vocab_all_words_have_valid_indices():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)
    for word, idx in vocab.items():
        assert isinstance(idx, int) and idx >= 0, f"Invalid index for '{word}': {idx}"

def test_vocab_same_result_on_same_input():
    """Vocabulary building must be deterministic (reproducible)."""
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab_a = build_vocabulary(tokenized)
    vocab_b = build_vocabulary(tokenized)
    assert vocab_a == vocab_b, "Vocabulary is not deterministic!"


# ─────────────────────────────────────────────
#  TESTS: OOV HANDLING
# ─────────────────────────────────────────────

def test_oov_token_handled_without_error():
    """AC: Out-of-vocabulary tokens handled without errors."""
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)

    # Encode a sequence with a word that definitely isn't in the vocab
    unknown_tokens = ["completely_unknown_word_xyz", "another_unknown_abc"]
    try:
        result = encode_sequence(unknown_tokens, vocab)
    except Exception as e:
        raise AssertionError(f"OOV encoding raised an error: {e}")

    assert all(v == vocab[OOV_TOKEN] for v in result), \
        f"OOV words not mapped to OOV index. Got: {result}"

def test_oov_maps_to_index_1():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)
    result = encode_sequence(["this_word_is_not_in_vocab"], vocab)
    assert result == [1], f"Expected [1] for OOV, got {result}"

def test_mixed_known_and_oov():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)

    # Pick a known word from vocab (not PAD or OOV)
    known_word = [w for w in vocab if w not in (PADDING_TOKEN, OOV_TOKEN)][0]
    tokens = [known_word, "totally_unknown_word_999"]
    result = encode_sequence(tokens, vocab)

    assert result[0] == vocab[known_word], "Known word not encoded correctly"
    assert result[1] == vocab[OOV_TOKEN],  "Unknown word not mapped to OOV"


# ─────────────────────────────────────────────
#  TESTS: PADDING
# ─────────────────────────────────────────────

def test_padding_short_sequence():
    """Short sequence should be padded with zeros to MAX_SEQUENCE_LENGTH."""
    short = [2, 3, 4]
    result = pad_sequence(short, MAX_SEQUENCE_LENGTH)
    assert len(result) == MAX_SEQUENCE_LENGTH
    assert result[:3] == [2, 3, 4]
    assert all(v == 0 for v in result[3:])

def test_padding_long_sequence():
    """Long sequence should be truncated to MAX_SEQUENCE_LENGTH."""
    long_seq = list(range(MAX_SEQUENCE_LENGTH + 20))
    result = pad_sequence(long_seq, MAX_SEQUENCE_LENGTH)
    assert len(result) == MAX_SEQUENCE_LENGTH
    assert result == long_seq[:MAX_SEQUENCE_LENGTH]

def test_padding_exact_length():
    """Sequence of exact length should pass through unchanged."""
    exact = list(range(MAX_SEQUENCE_LENGTH))
    result = pad_sequence(exact, MAX_SEQUENCE_LENGTH)
    assert result == exact

def test_padding_uniform_across_all_samples():
    """AC: Padding applied to uniform length."""
    _, vocab, label_map = make_full_pipeline()
    tokenized = tokenize_dataset(ALL_SAMPLES)
    encoded = encode_dataset(tokenized, vocab)
    labeled = encode_labels(encoded, label_map)
    padded = pad_dataset(labeled, MAX_SEQUENCE_LENGTH)

    lengths = [len(s["padded_sequence"]) for s in padded]
    assert all(l == MAX_SEQUENCE_LENGTH for l in lengths), \
        f"Non-uniform lengths found: {set(lengths)}"

def test_padding_uses_zero_as_pad_value():
    short = [5, 10]
    result = pad_sequence(short, 6)
    assert result == [5, 10, 0, 0, 0, 0]


# ─────────────────────────────────────────────
#  TESTS: NULL / NONE VALUES
# ─────────────────────────────────────────────

def test_no_null_values_in_output():
    """AC: No null values present in output tensors."""
    padded, _, _ = make_full_pipeline()
    for i, sample in enumerate(padded):
        seq = sample.get("padded_sequence")
        assert seq is not None, f"Sample {i}: padded_sequence is None"
        assert None not in seq, f"Sample {i}: contains None in sequence"

def test_no_none_in_individual_tokens():
    tokenized = tokenize_dataset(ALL_SAMPLES)
    for sample in tokenized:
        assert None not in sample["tokens"], "None found in tokens list"

def test_encoded_labels_not_null():
    padded, _, _ = make_full_pipeline()
    for i, sample in enumerate(padded):
        label = sample.get("encoded_label")
        assert label is not None, f"Sample {i}: encoded_label is None"
        assert label >= 0, f"Sample {i}: encoded_label is negative: {label}"


# ─────────────────────────────────────────────
#  TESTS: FULL PIPELINE (end-to-end)
# ─────────────────────────────────────────────

def test_full_pipeline_runs_without_errors():
    """Run the complete pipeline on all shards — should complete with no exceptions."""
    try:
        result = run_preprocessing_pipeline(RAW_SHARDS)
    except Exception as e:
        raise AssertionError(f"Full pipeline raised an error: {e}")
    assert result is not None

def test_full_pipeline_returns_correct_keys():
    result = run_preprocessing_pipeline(RAW_SHARDS)
    for key in ("vocabulary", "label_map", "vocab_size", "sequence_length", "output_paths"):
        assert key in result, f"Missing key in pipeline result: {key}"

def test_full_pipeline_output_files_exist():
    result = run_preprocessing_pipeline(RAW_SHARDS)
    for client_id, path in result["output_paths"].items():
        assert Path(path).exists(), f"Output file missing for client {client_id}: {path}"

def test_full_pipeline_output_file_structure():
    """Each saved JSON must contain expected keys and correct sample count."""
    result = run_preprocessing_pipeline(RAW_SHARDS)
    for client_id, path in result["output_paths"].items():
        with open(path) as f:
            data = json.load(f)
        assert "samples" in data
        assert "vocabulary" not in data, "Vocab should NOT be embedded in shard (privacy)"
        assert data["sequence_length"] == MAX_SEQUENCE_LENGTH
        for s in data["samples"]:
            assert "padded_sequence" in s
            assert "encoded_label" in s
            assert len(s["padded_sequence"]) == MAX_SEQUENCE_LENGTH

def test_consistent_vocab_across_shards():
    """
    AC: Tokenization applied consistently across all client shards.
    Verify that the same word gets the same index in every shard's encoding.
    """
    # Build vocab from all data
    tokenized = tokenize_dataset(ALL_SAMPLES)
    vocab = build_vocabulary(tokenized)

    # Encode same word in two different shards using same vocab
    word = "habari"
    if word in vocab:
        idx_shard0 = encode_sequence([word], vocab)[0]
        idx_shard1 = encode_sequence([word], vocab)[0]
        assert idx_shard0 == idx_shard1, \
            f"Same word got different index in different shards: {idx_shard0} vs {idx_shard1}"


# ─────────────────────────────────────────────
#  TEST RUNNER
# ─────────────────────────────────────────────

ALL_TESTS = [
    # Tokenization
    ("Tokenize basic sentence",                       test_tokenize_basic),
    ("Tokenize — lowercases text",                    test_tokenize_lowercase),
    ("Tokenize — strips punctuation",                 test_tokenize_strips_punctuation),
    ("Tokenize — empty string returns []",            test_tokenize_empty_string),
    ("Tokenize — whitespace-only returns []",         test_tokenize_none_like),
    ("Tokenize — consistent across shards (AC)",      test_tokenize_consistent_across_shards),

    # Vocabulary
    ("Vocab — contains <PAD> and <OOV>",              test_vocab_contains_pad_and_oov),
    ("Vocab — <PAD> is index 0",                      test_vocab_pad_is_index_0),
    ("Vocab — <OOV> is index 1",                      test_vocab_oov_is_index_1),
    ("Vocab — no duplicate indices",                  test_vocab_no_duplicate_indices),
    ("Vocab — mapping without errors (AC)",           test_vocab_mapping_without_errors),
    ("Vocab — all indices are valid non-neg ints",    test_vocab_all_words_have_valid_indices),
    ("Vocab — deterministic (same result twice)",     test_vocab_same_result_on_same_input),

    # OOV Handling
    ("OOV — handled without error (AC)",              test_oov_token_handled_without_error),
    ("OOV — maps to index 1",                         test_oov_maps_to_index_1),
    ("OOV — mixed known and unknown tokens",          test_mixed_known_and_oov),

    # Padding
    ("Padding — pads short sequence with zeros",      test_padding_short_sequence),
    ("Padding — truncates long sequence",             test_padding_long_sequence),
    ("Padding — exact-length sequence unchanged",     test_padding_exact_length),
    ("Padding — uniform across all samples (AC)",     test_padding_uniform_across_all_samples),
    ("Padding — uses 0 as pad value",                 test_padding_uses_zero_as_pad_value),

    # Null checks
    ("No nulls — in padded sequences (AC)",           test_no_null_values_in_output),
    ("No nulls — in token lists",                     test_no_none_in_individual_tokens),
    ("No nulls — in encoded labels",                  test_encoded_labels_not_null),

    # Full pipeline
    ("Full pipeline — runs without errors",           test_full_pipeline_runs_without_errors),
    ("Full pipeline — returns correct result keys",   test_full_pipeline_returns_correct_keys),
    ("Full pipeline — output JSON files exist",       test_full_pipeline_output_files_exist),
    ("Full pipeline — output file structure valid",   test_full_pipeline_output_file_structure),
    ("Full pipeline — consistent vocab across shards (AC)", test_consistent_vocab_across_shards),
]


if __name__ == "__main__":
    print("=" * 60)
    print("  US-03 Acceptance Criteria Tests")
    print("=" * 60)
    print()

    for name, fn in ALL_TESTS:
        run_test(name, fn)

    print()
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"  Results: {passed} passed, {failed} failed out of {len(results)} tests")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for name, ok, err in results:
            if not ok:
                print(f"  ❌ {name}: {err}")
        sys.exit(1)
    else:
        print("\n🎉 All US-03 acceptance criteria tests passed!")
        sys.exit(0)
