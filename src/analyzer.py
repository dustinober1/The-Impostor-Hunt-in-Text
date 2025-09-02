import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease, flesch_kincaid_grade
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt


class TextAuthenticityAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def load_data(self, train_dir, train_csv_path):
        '''Load training data from Kaggle structure: train/article_XXXX/file_1.txt, file_2.txt'''
        print(f"📖 Loading data from {train_dir} and {train_csv_path}")

        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"Training CSV not found: {train_csv_path}")

        train_df = pd.read_csv(train_csv_path)
        print(f"📊 Found {len(train_df)} training samples in CSV")

        texts_data = []
        missing_files = 0

        for idx, row in train_df.iterrows():
            article_id = row['id']
            real_text_id = row['real_text_id']

            # Create paths for Kaggle structure: train/article_XXXX/file_1.txt, file_2.txt
            article_dir = os.path.join(train_dir, f"article_{article_id:04d}")
            text_1_path = os.path.join(article_dir, "file_1.txt")
            text_2_path = os.path.join(article_dir, "file_2.txt")

            if os.path.exists(text_1_path) and os.path.exists(text_2_path):
                try:
                    with open(text_1_path, 'r', encoding='utf-8') as f:
                        text_1 = f.read().strip()
                    with open(text_2_path, 'r', encoding='utf-8') as f:
                        text_2 = f.read().strip()

                    texts_data.append({
                        'article_id': article_id,
                        'text_1': text_1,
                        'text_2': text_2,
                        'real_text_id': real_text_id,
                        'real_text': text_1 if real_text_id == 1 else text_2,
                        'fake_text': text_2 if real_text_id == 1 else text_1
                    })
                except Exception as e:
                    print(f"❌ Error reading files for article {article_id}: {e}")
                    missing_files += 1
            else:
                print(f"❌ Missing files for article {article_id} at {article_dir}")
                missing_files += 1

                if idx < 5:  # Show details for first few missing files
                    print(f"  Expected: {text_1_path}")
                    print(f"  Exists: {os.path.exists(text_1_path)}")
                    print(f"  Expected: {text_2_path}")
                    print(f"  Exists: {os.path.exists(text_2_path)}")

        if missing_files > 0:
            print(f"⚠️  {missing_files} files could not be loaded")

        df = pd.DataFrame(texts_data)
        print(f"✅ Successfully loaded {len(df)} text pairs")
        return df

    def extract_linguistic_features(self, text):
        '''Extract comprehensive linguistic features from text'''
        features = {}

        if not text or len(text.strip()) == 0:
            # Return default features for empty text
            return {f: 0 for f in ['length', 'word_count', 'sentence_count', 'avg_word_length',
                                  'avg_sentence_length', 'flesch_reading_ease', 'flesch_kincaid_grade',
                                  'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound',
                                  'exclamation_count', 'question_count', 'comma_count', 'semicolon_count',
                                  'colon_count', 'quotation_count', 'capital_ratio', 'digit_ratio',
                                  'most_frequent_word_ratio', 'unique_word_ratio', 'space_term_ratio']}

        # Basic statistics
        features['length'] = len(text)
        words = text.split()
        features['word_count'] = len(words)
        sentences = re.split(r'[.!?]+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        features['sentence_count'] = len(sentences) if sentences else 1

        if words:
            features['avg_word_length'] = np.mean([len(word) for word in words])
        else:
            features['avg_word_length'] = 0

        features['avg_sentence_length'] = features['word_count'] / max(features['sentence_count'], 1)

        # Readability scores (handle potential errors)
        try:
            features['flesch_reading_ease'] = flesch_reading_ease(text)
        except:
            features['flesch_reading_ease'] = 0

        try:
            features['flesch_kincaid_grade'] = flesch_kincaid_grade(text)
        except:
            features['flesch_kincaid_grade'] = 0

        # Sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        features['sentiment_pos'] = sentiment['pos']
        features['sentiment_neg'] = sentiment['neg']
        features['sentiment_neu'] = sentiment['neu']
        features['sentiment_compound'] = sentiment['compound']

        # Punctuation and special characters
        features['exclamation_count'] = text.count('!')
        features['question_count'] = text.count('?')
        features['comma_count'] = text.count(',')
        features['semicolon_count'] = text.count(';')
        features['colon_count'] = text.count(':')
        features['quotation_count'] = text.count('"') + text.count("'")

        # Capital letters and formatting
        total_chars = len(text)
        if total_chars > 0:
            features['capital_ratio'] = sum(1 for c in text if c.isupper()) / total_chars
            features['digit_ratio'] = sum(1 for c in text if c.isdigit()) / total_chars
        else:
            features['capital_ratio'] = 0
            features['digit_ratio'] = 0

        # Word frequency patterns (potential hallucination indicators)
        if words:
            words_lower = [w.lower() for w in words]
            word_freq = Counter(words_lower)
            features['most_frequent_word_ratio'] = max(word_freq.values()) / len(words)
            features['unique_word_ratio'] = len(set(words_lower)) / len(words)
        else:
            features['most_frequent_word_ratio'] = 0
            features['unique_word_ratio'] = 0

        # Space-related terms (domain-specific)
        space_terms = ['space', 'satellite', 'orbit', 'mission', 'spacecraft', 'astronaut',
                      'telescope', 'planet', 'solar', 'cosmic', 'research', 'experiment',
                      'esa', 'nasa', 'observatory', 'launch', 'rocket', 'mars', 'earth',
                      'astronomy', 'astrophysics', 'galaxy', 'universe', 'station']

        if words:
            words_lower = [w.lower() for w in words]
            space_term_count = sum(1 for word in words_lower
                                 if any(term in word for term in space_terms))
            features['space_term_ratio'] = space_term_count / len(words)
        else:
            features['space_term_ratio'] = 0

        return features

    def extract_semantic_features(self, text1, text2):
        '''Extract features comparing two texts'''
        features = {}

        if not text1 or not text2:
            return {'tfidf_similarity': 0, 'word_overlap_ratio': 0,
                   'length_diff': 0, 'length_ratio': 0}

        # TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            features['tfidf_similarity'] = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            features['tfidf_similarity'] = 0

        # Word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if words1 or words2:
            features['word_overlap_ratio'] = len(words1.intersection(words2)) / len(words1.union(words2))
        else:
            features['word_overlap_ratio'] = 0

        # Length comparison
        len1, len2 = len(text1), len(text2)
        features['length_diff'] = abs(len1 - len2)
        if max(len1, len2) > 0:
            features['length_ratio'] = min(len1, len2) / max(len1, len2)
        else:
            features['length_ratio'] = 1

        return features

    def analyze_dataset(self, df):
        '''Perform comprehensive analysis of the dataset'''
        print("🔍 Analyzing dataset features...")

        if len(df) == 0:
            print("❌ No data to analyze!")
            return None, None, None, None

        # Extract features for all texts
        real_features = []
        fake_features = []
        pair_features = []

        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"  Processing sample {idx+1}/{len(df)}")

            real_feat = self.extract_linguistic_features(row['real_text'])
            fake_feat = self.extract_linguistic_features(row['fake_text'])
            pair_feat = self.extract_semantic_features(row['real_text'], row['fake_text'])

            real_features.append({**real_feat, 'type': 'real'})
            fake_features.append({**fake_feat, 'type': 'fake'})
            pair_features.append(pair_feat)

        # Convert to DataFrames
        real_df = pd.DataFrame(real_features)
        fake_df = pd.DataFrame(fake_features)
        all_features_df = pd.concat([real_df, fake_df], ignore_index=True)
        pair_df = pd.DataFrame(pair_features)

        print("✅ Feature extraction complete!")
        return real_df, fake_df, all_features_df, pair_df

    def plot_feature_comparison(self, real_df, fake_df, figsize=(20, 12)):
        '''Plot comparison between real and fake text features'''
        if real_df is None or fake_df is None or len(real_df) == 0:
            print("❌ No data to plot!")
            return

        feature_cols = ['length', 'word_count', 'avg_word_length', 'flesch_reading_ease',
                       'sentiment_compound', 'unique_word_ratio', 'space_term_ratio',
                       'capital_ratio']

        fig, axes = plt.subplots(2, 4, figsize=figsize)
        axes = axes.flatten()

        for i, feature in enumerate(feature_cols):
            if i < len(axes) and feature in real_df.columns:
                ax = axes[i]

                # Create histograms
                ax.hist(real_df[feature], alpha=0.7, label='Real', bins=20, color='blue', density=True)
                ax.hist(fake_df[feature], alpha=0.7, label='Fake', bins=20, color='red', density=True)

                ax.set_xlabel(feature.replace('_', ' ').title())
                ax.set_ylabel('Density')
                ax.set_title(f'Distribution of {feature.replace("_", " ").title()}')
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Statistical comparison
        print("\n📈 Statistical Comparison (Real vs Fake):")
        print("=" * 60)
        for feature in feature_cols:
            if feature in real_df.columns:
                real_mean = real_df[feature].mean()
                fake_mean = fake_df[feature].mean()
                diff_pct = ((real_mean - fake_mean) / fake_mean * 100) if fake_mean != 0 else 0
                print(f"{feature:20s} | Real: {real_mean:8.3f} | Fake: {fake_mean:8.3f} | Diff: {diff_pct:6.1f}%")
