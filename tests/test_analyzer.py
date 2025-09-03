
import pytest
import pandas as pd
from src.analyzer import TextAuthenticityAnalyzer

@pytest.fixture
def analyzer():
    return TextAuthenticityAnalyzer()

def test_extract_linguistic_features_empty_text(analyzer):
    features = analyzer.extract_linguistic_features("")
    assert isinstance(features, dict)
    for value in features.values():
        assert value == 0

def test_extract_linguistic_features_simple_text(analyzer):
    text = "This is a simple sentence. It has two sentences."
    features = analyzer.extract_linguistic_features(text)
    assert features['length'] == 48
    assert features['word_count'] == 9
    assert features['sentence_count'] == 2
    assert features['avg_word_length'] > 0
    assert features['avg_sentence_length'] > 0

def test_extract_semantic_features_empty_texts(analyzer):
    features = analyzer.extract_semantic_features("", "")
    assert isinstance(features, dict)
    for value in features.values():
        assert value == 0 or value == 1 # length_ratio is 1

def test_extract_semantic_features_simple_texts(analyzer):
    text1 = "This is the first text."
    text2 = "This is the second text."
    features = analyzer.extract_semantic_features(text1, text2)
    assert features['tfidf_similarity'] > 0
    assert features['word_overlap_ratio'] > 0
    assert features['length_diff'] > 0
    assert features['length_ratio'] > 0

def test_analyze_dataset(analyzer):
    data = {
        'real_text': ["This is a real text."],
        'fake_text': ["This is a fake text."]
    }
    df = pd.DataFrame(data)
    real_df, fake_df, all_features_df, pair_df = analyzer.analyze_dataset(df)
    assert not real_df.empty
    assert not fake_df.empty
    assert not all_features_df.empty
    assert not pair_df.empty

