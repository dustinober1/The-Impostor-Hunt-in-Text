
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.classifier import TextAuthenticityClassifier
from src.analyzer import TextAuthenticityAnalyzer

@pytest.fixture
def classifier():
    return TextAuthenticityClassifier()

@pytest.fixture
def mock_analyzer():
    analyzer = MagicMock(spec=TextAuthenticityAnalyzer)
    analyzer.extract_linguistic_features.return_value = {'length': 10, 'word_count': 2}
    analyzer.extract_semantic_features.return_value = {'tfidf_similarity': 0.5}
    return analyzer

def test_prepare_training_data(classifier, mock_analyzer):
    classifier.analyzer = mock_analyzer
    data = {
        'text_1': ["This is text 1."],
        'text_2': ["This is text 2."],
        'real_text_id': [1]
    }
    df = pd.DataFrame(data)
    X, y = classifier.prepare_training_data(df)
    assert not X.empty
    assert y.shape == (1,)
    assert 'length_1' in X.columns
    assert 'word_count_2' in X.columns
    assert 'tfidf_similarity' in X.columns

