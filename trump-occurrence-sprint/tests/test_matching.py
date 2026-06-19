from src.matching import MatchConfig, first_occurrence, occurrences, tokenize


def test_single_words_casing_and_boundaries():
    tokens = tokenize("War is not toward or warmer. WAR ends.")
    assert occurrences("war", tokens) == [0, 6]


def test_phrase_and_repeated_whitespace():
    tokens = tokenize("Secure   America, Act. Secure America Act again.")
    assert occurrences("secure america act", tokens) == [0, 3]


def test_hyphenation_default_split():
    tokens = tokenize("The border-wall plan mentions border wall twice.")
    assert occurrences("border wall", tokens) == [1, 5]


def test_hyphenation_configurable_no_split():
    cfg = MatchConfig(split_hyphens=False)
    tokens = tokenize("border-wall", split_hyphens=False)
    assert occurrences("border wall", tokens, cfg) == []


def test_possessives():
    tokens = tokenize("Trump's policy and Trump policy.")
    assert occurrences("trump", tokens) == [0, 3]


def test_target_at_start_and_end():
    tokens = tokenize("America first and then America")
    assert first_occurrence("america", tokens) == 0
    assert occurrences("america", tokens) == [0, 4]


def test_punctuation_phrase_boundary():
    tokens = tokenize('"New York," he said. New York!')
    assert occurrences("new york", tokens) == [0, 4]
