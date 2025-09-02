import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
from src.analyzer import TextAuthenticityAnalyzer


class TextAuthenticityClassifier:
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.models = {}
        self.ensemble_model = None
        self.feature_names = None

    def prepare_training_data(self, df):
        '''Prepare training data with features for both texts in each pair'''
        print("🛠️  Preparing training data...")

        if len(df) == 0:
            print("❌ No training data available!")
            return None, None

        X = []
        y = []

        analyzer = TextAuthenticityAnalyzer()

        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"  Processing sample {idx+1}/{len(df)}")

            # Extract features for both texts
            feat_1 = analyzer.extract_linguistic_features(row['text_1'])
            feat_2 = analyzer.extract_linguistic_features(row['text_2'])
            pair_feat = analyzer.extract_semantic_features(row['text_1'], row['text_2'])

            # Create feature difference and ratio vectors
            feature_diff = {}
            for key in feat_1.keys():
                if key in feat_2:
                    feature_diff[f'{key}_diff'] = feat_1[key] - feat_2[key]
                    if feat_2[key] != 0:
                        feature_diff[f'{key}_ratio'] = feat_1[key] / feat_2[key]
                    else:
                        feature_diff[f'{key}_ratio'] = feat_1[key]

            # Combine all features
            combined_features = {
                **{f'{k}_1': v for k, v in feat_1.items()},
                **{f'{k}_2': v for k, v in feat_2.items()},
                **feature_diff,
                **pair_feat
            }

            X.append(combined_features)
            y.append(1 if row['real_text_id'] == 1 else 0)  # 1 if text_1 is real, 0 if text_2 is real

        # Convert to DataFrame
        X_df = pd.DataFrame(X)
        X_df = X_df.fillna(0)  # Fill NaN values
        X_df = X_df.replace([np.inf, -np.inf], 0)  # Replace infinite values

        self.feature_names = X_df.columns.tolist()
        print(f"✅ Created {len(self.feature_names)} features")

        return X_df, np.array(y)

    def prepare_test_data(self, test_dir):
        '''Prepare test data from Kaggle structure: test/article_XXXX/file_1.txt, file_2.txt'''
        print(f"🧪 Preparing test data from {test_dir}...")

        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")

        # Get all article subdirectories
        article_dirs = [d for d in os.listdir(test_dir)
                       if os.path.isdir(os.path.join(test_dir, d)) and d.startswith('article_')]

        # Extract article IDs and sort
        article_ids = []
        for article_dir in article_dirs:
            try:
                article_id = int(article_dir.replace('article_', ''))
                article_ids.append(article_id)
            except ValueError:
                continue

        article_ids = sorted(article_ids)
        print(f"📊 Found {len(article_ids)} test samples")

        X_test = []
        test_info = []

        analyzer = TextAuthenticityAnalyzer()

        for i, article_id in enumerate(article_ids):
            if i % 10 == 0:
                print(f"  Processing test sample {i+1}/{len(article_ids)}")

            article_dir = os.path.join(test_dir, f"article_{article_id:04d}")
            text_1_path = os.path.join(article_dir, "file_1.txt")
            text_2_path = os.path.join(article_dir, "file_2.txt")

            if os.path.exists(text_1_path) and os.path.exists(text_2_path):
                try:
                    with open(text_1_path, 'r', encoding='utf-8') as f:
                        text_1 = f.read().strip()
                    with open(text_2_path, 'r', encoding='utf-8') as f:
                        text_2 = f.read().strip()

                    # Extract features
                    feat_1 = analyzer.extract_linguistic_features(text_1)
                    feat_2 = analyzer.extract_linguistic_features(text_2)
                    pair_feat = analyzer.extract_semantic_features(text_1, text_2)

                    # Create feature difference and ratio vectors
                    feature_diff = {}
                    for key in feat_1.keys():
                        if key in feat_2:
                            feature_diff[f'{key}_diff'] = feat_1[key] - feat_2[key]
                            if feat_2[key] != 0:
                                feature_diff[f'{key}_ratio'] = feat_1[key] / feat_2[key]
                            else:
                                feature_diff[f'{key}_ratio'] = feat_1[key]

                    # Combine all features
                    combined_features = {
                        **{f'{k}_1': v for k, v in feat_1.items()},
                        **{f'{k}_2': v for k, v in feat_2.items()},
                        **feature_diff,
                        **pair_feat
                    }

                    X_test.append(combined_features)
                    test_info.append(article_id)

                except Exception as e:
                    print(f"❌ Error processing test sample {article_id}: {e}")

        X_test_df = pd.DataFrame(X_test)

        # Ensure same columns as training data
        if self.feature_names:
            for col in self.feature_names:
                if col not in X_test_df.columns:
                    X_test_df[col] = 0
            X_test_df = X_test_df[self.feature_names]

        X_test_df = X_test_df.fillna(0)
        X_test_df = X_test_df.replace([np.inf, -np.inf], 0)

        print(f"✅ Test data prepared: {X_test_df.shape}")
        return X_test_df, test_info

    def train_models(self, X, y, cv_folds=5):
        '''Train multiple models with cross-validation'''
        print("🤖 Training models...")
        print(f"📊 Training data shape: {X.shape}")
        print(f"🎯 Class distribution: {np.bincount(y)}")

        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # Feature selection
        n_features = min(50, X_train.shape[1])  # Select top features
        self.feature_selector = SelectKBest(f_classif, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_val_selected = self.feature_selector.transform(X_val_scaled)

        print(f"🔍 Selected {n_features} best features")

        # Define models
        models_config = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=42, eval_metric='logloss'
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=42, verbose=-1
            ),
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, C=1.0
            )
        }

        # Train and evaluate individual models
        model_scores = {}

        for name, model in models_config.items():
            print(f"🔧 Training {name}...")

            # Cross-validation
            try:
                cv_scores = cross_val_score(
                    model, X_train_selected, y_train, cv=cv_folds,
                    scoring='accuracy', n_jobs=-1
                )

                # Train on full training set
                model.fit(X_train_selected, y_train)

                # Validation performance
                val_pred = model.predict(X_val_selected)
                val_acc = accuracy_score(y_val, val_pred)

                model_scores[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'val_acc': val_acc
                }

                print(f"  CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                print(f"  Val Accuracy: {val_acc:.4f}")

                self.models[name] = model

            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")

        if not self.models:
            print("❌ No models trained successfully!")
            return 0, {}

        # Create ensemble
        estimators = [(name.replace(' ', '_'), model) for name, model in self.models.items()]
        self.ensemble_model = VotingClassifier(estimators=estimators, voting='soft')

        print("🎯 Training ensemble model...")
        self.ensemble_model.fit(X_train_selected, y_train)

        # Evaluate ensemble
        ensemble_pred = self.ensemble_model.predict(X_val_selected)
        ensemble_acc = accuracy_score(y_val, ensemble_pred)

        print(f"🏆 Ensemble validation accuracy: {ensemble_acc:.4f}")

        # Show model comparison
        print("\n📊 Model Performance Summary:")
        print("=" * 60)
        for name, scores in model_scores.items():
            print(f"{name:20s} | CV: {scores['cv_mean']:.4f} | Val: {scores['val_acc']:.4f}")
        print(f"{'Ensemble':20s} | Val: {ensemble_acc:.4f}")

        return ensemble_acc, model_scores

    def predict(self, X_test):
        '''Make predictions on test data'''
        print("🔮 Making predictions...")

        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)

        # Get ensemble predictions
        predictions = self.ensemble_model.predict(X_test_selected)
        probabilities = self.ensemble_model.predict_proba(X_test_selected)

        print(f"✅ Generated {len(predictions)} predictions")
        return predictions, probabilities

    def create_submission(self, test_dir, output_file='submission.csv'):
        '''Create submission file'''
        print(f"📝 Creating submission file: {output_file}")

        X_test, test_info = self.prepare_test_data(test_dir)
        predictions, probabilities = self.predict(X_test)

        submission_data = []
        for i, article_id in enumerate(test_info):
            # predictions[i] = 1 means text_1 is real, 0 means text_2 is real
            real_text_id = 1 if predictions[i] == 1 else 2
            confidence = max(probabilities[i])

            submission_data.append({
                'id': article_id,
                'real_text_id': real_text_id,
                'confidence': confidence
            })

        submission_df = pd.DataFrame(submission_data)
        submission_df = submission_df.sort_values('id')

        # Save submission file (only required columns)
        final_submission = submission_df[['id', 'real_text_id']].copy()
        final_submission.to_csv(output_file, index=False)

        print(f"✅ Submission saved to {output_file}")
        print(f"📊 Prediction distribution: {submission_df['real_text_id'].value_counts().to_dict()}")
        print(f"📈 Average confidence: {submission_df['confidence'].mean():.4f}")

        return submission_df
