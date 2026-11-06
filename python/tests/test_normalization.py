from hinglish.normalization import NormalizationConfig, normalize_text


def test_url_and_mention_masking():
    cfg = NormalizationConfig()
    text = "check this out @someuser http://example.com/x amazing!!! 😀"
    out = normalize_text(text, cfg)
    assert "<URL>" in out
    assert "<USER>" in out
    assert "example.com" not in out
    assert "@someuser" not in out


def test_emoji_and_repeats_and_punctuation_preserved():
    cfg = NormalizationConfig()
    text = "sooooo goooood!!! 😍😍"
    out = normalize_text(text, cfg)
    assert "sooooo" in out
    assert "goooood" in out
    assert "!!!" in out
    assert "😍" in out


def test_lowercase_toggle():
    cfg_on = NormalizationConfig(lowercase=True)
    cfg_off = NormalizationConfig(lowercase=False)
    text = "This Is AMAZING"
    assert normalize_text(text, cfg_on) == "this is amazing"
    assert normalize_text(text, cfg_off) == "This Is AMAZING"


def test_whitespace_collapsing():
    cfg = NormalizationConfig()
    out = normalize_text("hello    world\n\tfoo", cfg)
    assert out == "hello world foo"
