"""
Machine Learning Engine for Fake News Detection.
Implements TF-IDF vectorization, multi-model training (PassiveAggressive,
LogisticRegression, NaiveBayes, RandomForest), model comparison metrics,
token-level Explainable AI (XAI), and sensationalism/objectivity scoring.
"""

import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

SENSATIONAL_WORDS = {
    "shocking", "secret", "miracle", "cures", "instantly", "banned", "unbelievable",
    "elites", "cabal", "conspiracy", "suppressed", "microchips", "mind control",
    "leaked", "whistleblower", "wiped", "urgent", "warning", "you won't believe",
    "click now", "deleted forever", "clone", "darkness", "bribed", "amnesia",
    "alien", "pyramid", "confiscate", "time traveler", "10,000 times", "chemotherapy",
    "wiretap", "sphere", "mutates", "carcinogens", "disappear", "bizarre", "forbidden"
}

RELIABLE_INDICATORS = {
    "announced", "reported", "officials", "according", "researchers", "study",
    "published", "data", "conference", "agreement", "reiterated", "demonstrated",
    "analysts", "survey", "ministers", "legislation", "spokesperson", "scientific",
    "journal", "trial", "peer-reviewed", "department", "bureau", "statistics"
}

class FakeNewsMLEngine:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            sublinear_tf=True
        )
        self.models = {
            "passive_aggressive": SGDClassifier(loss='log_loss', penalty='l2', max_iter=200, random_state=42),
            "logistic_regression": LogisticRegression(max_iter=200, random_state=42, C=1.0),
            "naive_bayes": MultinomialNB(alpha=0.5),
            "random_forest": RandomForestClassifier(n_estimators=50, random_state=42)
        }
        self.model_names = {
            "passive_aggressive": "Passive Aggressive Classifier",
            "logistic_regression": "Logistic Regression",
            "naive_bayes": "Multinomial Naive Bayes",
            "random_forest": "Random Forest Classifier",
            "ensemble": "Ensemble Soft Voting Model"
        }
        self.metrics = {}
        self.is_trained = False

    def clean_text(self, text):
        """Preprocesses text by removing special characters and standardizing whitespace."""
        if not text:
            return ""
        text = str(text).strip()
        # Keep punctuation for sensationalism metric calculation later, but clean for ML
        clean = text.lower()
        clean = re.sub(r'https?://\S+|www\.\S+', '', clean)
        clean = re.sub(r'<.*?>+', '', clean)
        clean = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean

    def train(self, df):
        """Trains all models on the provided DataFrame."""
        df['clean_full_text'] = df['full_text'].apply(self.clean_text)
        
        X = self.vectorizer.fit_transform(df['clean_full_text'])
        y = df['label'].values

        self.metrics = {}
        
        for key, model in self.models.items():
            model.fit(X, y)
            preds = model.predict(X)
            
            # Predict probabilities if supported, else decision function
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                df_val = model.decision_function(X)
                probs = 1 / (1 + np.exp(-df_val))
            else:
                probs = preds.astype(float)

            acc = accuracy_score(y, preds)
            prec = precision_score(y, preds, zero_division=0)
            rec = recall_score(y, preds, zero_division=0)
            f1 = f1_score(y, preds, zero_division=0)
            cm = confusion_matrix(y, preds).tolist()

            self.metrics[key] = {
                "name": self.model_names[key],
                "accuracy": round(float(acc) * 100, 2),
                "precision": round(float(prec) * 100, 2),
                "recall": round(float(rec) * 100, 2),
                "f1_score": round(float(f1) * 100, 2),
                "confusion_matrix": cm # [[TN, FP], [FN, TP]]
            }

        # Ensemble Metrics
        ensemble_probs = self._get_ensemble_probs(X)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        
        acc = accuracy_score(y, ensemble_preds)
        prec = precision_score(y, ensemble_preds, zero_division=0)
        rec = recall_score(y, ensemble_preds, zero_division=0)
        f1 = f1_score(y, ensemble_preds, zero_division=0)
        cm = confusion_matrix(y, ensemble_preds).tolist()

        self.metrics["ensemble"] = {
            "name": self.model_names["ensemble"],
            "accuracy": round(float(acc) * 100, 2),
            "precision": round(float(prec) * 100, 2),
            "recall": round(float(rec) * 100, 2),
            "f1_score": round(float(f1) * 100, 2),
            "confusion_matrix": cm
        }

        self.is_trained = True
        return self.metrics

    def _get_ensemble_probs(self, X):
        """Calculates soft voting average probability across trained models."""
        probs_list = []
        for model in self.models.values():
            if hasattr(model, "predict_proba"):
                probs_list.append(model.predict_proba(X)[:, 1])
            elif hasattr(model, "decision_function"):
                df_val = model.decision_function(X)
                sig = 1 / (1 + np.exp(-df_val))
                probs_list.append(sig)
            else:
                preds = model.predict(X)
                probs_list.append(preds.astype(float))
        return np.mean(probs_list, axis=0)

    def calculate_sensationalism_score(self, text):
        """Calculates sensationalism index (0-100%) based on text stylistic features."""
        if not text:
            return 0.0
        words = re.findall(r'\b\w+\b', text)
        if not words:
            return 0.0
            
        all_caps_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        lowered_text = text.lower()
        sensational_hit = sum(1 for w in SENSATIONAL_WORDS if w in lowered_text)
        
        # Calculate ratio
        caps_ratio = min((all_caps_count / len(words)) * 100, 40)
        punc_impact = min((exclamation_count * 10 + question_count * 5), 30)
        keyword_impact = min(sensational_hit * 15, 50)
        
        score = min(caps_ratio + punc_impact + keyword_impact, 100.0)
        return round(score, 1)

    def calculate_objectivity_score(self, text):
        """Calculates objectivity rating (0-100%). High objectivity = low bias."""
        if not text:
            return 50.0
        lowered_text = text.lower()
        reliable_hits = sum(1 for w in RELIABLE_INDICATORS if w in lowered_text)
        sensational_hits = sum(1 for w in SENSATIONAL_WORDS if w in lowered_text)
        
        words = re.findall(r'\b\w+\b', lowered_text)
        total_words = len(words)
        
        base = 60.0
        base += min(reliable_hits * 10, 30)
        base -= min(sensational_hits * 15, 40)
        
        # Longer well-structured articles tend to be more objective than tiny snippets
        if total_words > 40:
            base += 10
        elif total_words < 15:
            base -= 10
            
        return round(float(np.clip(base, 5.0, 98.0)), 1)

    def calculate_readability_score(self, text):
        """Computes basic Flesch Reading Ease score estimate (0-100)."""
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s for s in sentences if s.strip()]
        
        num_words = max(len(words), 1)
        num_sentences = max(len(sentences), 1)
        
        # Approximate syllable count by vowel groups
        syllable_count = 0
        for word in words:
            word_syllables = len(re.findall(r'[aeiouyAEIOUY]+', word))
            syllable_count += max(word_syllables, 1)
            
        score = 206.835 - (1.015 * (num_words / num_sentences)) - (84.6 * (syllable_count / num_words))
        return round(float(np.clip(score, 10.0, 100.0)), 1)

    def predict(self, text, model_key="ensemble"):
        """
        Predicts news authenticity and returns credibility score, XAI tokens, and analytical metrics.
        """
        if not self.is_trained:
            raise ValueError("Model engine is not trained yet.")
            
        clean_t = self.clean_text(text)
        if not clean_t:
            return {
                "error": "Empty text provided."
            }

        vec = self.vectorizer.transform([clean_t])

        if model_key == "ensemble" or model_key not in self.models:
            prob_real = float(self._get_ensemble_probs(vec)[0])
        else:
            model = self.models[model_key]
            if hasattr(model, "predict_proba"):
                prob_real = float(model.predict_proba(vec)[0][1])
            elif hasattr(model, "decision_function"):
                df_val = float(model.decision_function(vec)[0])
                prob_real = float(1 / (1 + np.exp(-df_val)))
            else:
                prob_real = float(model.predict(vec)[0])

        # prob_real represents probability of REAL NEWS (1)
        credibility_score = round(prob_real * 100, 1)
        fake_probability = round((1 - prob_real) * 100, 1)
        
        if credibility_score >= 65:
            verdict = "REAL"
            confidence_label = "HIGHLY RELIABLE"
        elif credibility_score >= 45:
            verdict = "NEUTRAL / UNVERIFIED"
            confidence_label = "MIXED SIGNALS"
        else:
            verdict = "FAKE"
            confidence_label = "HIGHLY SUSPICIOUS"

        # Explainable AI (XAI) Token Scoring
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        # Use Logistic Regression coefficients for feature weight interpretability
        lr_model = self.models["logistic_regression"]
        coefs = lr_model.coef_[0] # positive = Real indicator, negative = Fake indicator

        words_in_original = re.findall(r'\b\w+\b|[^\w\s]', text)
        token_highlights = []
        
        for token in words_in_original:
            token_clean = token.lower().strip()
            score = 0.0
            label = "NEUTRAL"
            
            if token_clean in feature_names:
                idx = np.where(feature_names == token_clean)[0]
                if len(idx) > 0:
                    weight = float(coefs[idx[0]])
                    score = round(weight, 3)
                    if weight > 0.05:
                        label = "REAL_INDICATOR"
                    elif weight < -0.05:
                        label = "FAKE_INDICATOR"

            # Check rule-based override for known sensationalist / reliable terms
            if token_clean in SENSATIONAL_WORDS:
                label = "FAKE_INDICATOR"
                score = -0.8
            elif token_clean in RELIABLE_INDICATORS:
                label = "REAL_INDICATOR"
                score = 0.8

            token_highlights.append({
                "token": token,
                "score": score,
                "label": label
            })

        sensationalism = self.calculate_sensationalism_score(text)
        objectivity = self.calculate_objectivity_score(text)
        readability = self.calculate_readability_score(text)

        return {
            "model_used": self.model_names.get(model_key, "Ensemble Model"),
            "verdict": verdict,
            "confidence_label": confidence_label,
            "credibility_score": credibility_score,
            "fake_probability": fake_probability,
            "sensationalism_score": sensationalism,
            "objectivity_score": objectivity,
            "readability_score": readability,
            "word_count": len(words_in_original),
            "token_highlights": token_highlights
        }
