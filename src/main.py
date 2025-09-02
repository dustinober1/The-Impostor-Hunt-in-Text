#!/usr/bin/env python
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Text analysis
import nltk

# Download NLTK data
try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')

from src.analyzer import TextAuthenticityAnalyzer
from src.classifier import TextAuthenticityClassifier


BASE_PATH = '/kaggle/input/fake-or-real-the-impostor-hunt/data'
TRAIN_CSV = f'{BASE_PATH}/train.csv'
TRAIN_DIR = f'{BASE_PATH}/train'  # Contains article_XXXX subdirectories
TEST_DIR = f'{BASE_PATH}/test'    # Contains article_XXXX subdirectories
SUBMISSION_FILE = 'submission.csv'

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def main():
    '''Main execution function for the complete pipeline'''
    print("🚀 Starting Text Authenticity Detection Pipeline")
    print("=" * 60)

    # Initialize analyzer
    analyzer = TextAuthenticityAnalyzer()

    # Load training data
    try:
        df = analyzer.load_data(TRAIN_DIR, TRAIN_CSV)
        if len(df) == 0:
            print("❌ No training data loaded. Please check the file structure.")
            return
        print(f"📚 Loaded {len(df)} training samples")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Display sample data
    print("\n📋 Sample Data:")
    sample_row = df.iloc[0]
    print(f"Article ID: {sample_row['article_id']}")
    print(f"Real text is: file_{sample_row['real_text_id']}.txt")
    print(f"Real text preview (first 200 chars): {sample_row['real_text'][:200]}...")
    print(f"Fake text preview (first 200 chars): {sample_row['fake_text'][:200]}...")

    # Analyze dataset
    print("\n" + "="*60)
    real_df, fake_df, all_df, pair_df = analyzer.analyze_dataset(df)

    if real_df is not None:
        # Plot comparisons
        print("\n📊 Plotting feature distributions...")
        analyzer.plot_feature_comparison(real_df, fake_df)

    # Train classifier
    print("\n" + "="*60)
    classifier = TextAuthenticityClassifier()

    # Prepare training data
    X, y = classifier.prepare_training_data(df)

    if X is not None and y is not None:
        # Train models
        ensemble_acc, model_scores = classifier.train_models(X, y)

        # Create submission
        print("\n" + "="*60)
        try:
            submission_df = classifier.create_submission(TEST_DIR, SUBMISSION_FILE)

            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📁 Submission file created: {SUBMISSION_FILE}")
            print(f"🏆 Best validation accuracy: {ensemble_acc:.4f}")

        except Exception as e:
            print(f"❌ Error creating submission: {e}")
            print("This might be because the test directory doesn't exist or is empty.")
            print("The model is trained and ready to use when test data becomes available.")
    else:
        print("❌ Could not prepare training data. Check the data loading process.")

if __name__ == "__main__":
    main()