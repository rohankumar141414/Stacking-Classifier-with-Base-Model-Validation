
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class StackedModel:
    def __init__(self, base_models, corr_threshold=0.9, cv=5, random_state=42, passthrough=False):
        self.base_models = base_models
        self.corr_threshold = corr_threshold
        self.cv = cv
        self.random_state = random_state
        self.passthrough = passthrough
        self.selected_models = None
        self.stacker = None

    def _select_models(self, X, y):
        pred_probas = {}
        for name, model in self.base_models:
            preds = cross_val_predict(model, X, y, cv=self.cv, method='predict_proba')[:, 1]
            pred_probas[name] = preds

        pred_df = pd.DataFrame(pred_probas)
        corr_matrix = pred_df.corr(method='spearman')
        print("Spearman Correlation Matrix:\n", corr_matrix)

        performance = {}
        for name, model in self.base_models:
            scores = cross_val_score(model, X, y, cv=self.cv, scoring='roc_auc', n_jobs=-1)
            performance[name] = scores.mean()
        print("\nBase Learner Performance:")
        for name, score in performance.items():
            print(f"{name.upper():>4}: {score:.3f}")

        to_drop = set()
        for i, j in zip(*np.where(np.triu(np.abs(corr_matrix) > self.corr_threshold, k=1))):
            name_i = corr_matrix.index[i]
            name_j = corr_matrix.columns[j]
            drop = name_i if performance[name_i] < performance[name_j] else name_j
            to_drop.add(drop)
        print(f"\nModels to drop due to high correlation (>{self.corr_threshold}): {to_drop}")

        selected = {name: model for name, model in self.base_models if name not in to_drop}
        print(f"\nSelected Models ({len(selected)}/{len(self.base_models)}):")
        print(f"\n{list(selected.keys())}")

        self.selected_models = list(selected.items())

    def fit(self, X, y):
        self._select_models(X, y)
        self.stacker = StackingClassifier(
            estimators=self.selected_models,
            cv=self.cv,
            passthrough=self.passthrough,
            n_jobs=-1
        )
        # print("Training stacking ensemble…")
        self.stacker.fit(X, y)
        # y_pred = self.stacker.predict(X)

        # cm = confusion_matrix(y, y_pred)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.stacker.classes_)
        # disp.plot(cmap='Blues')
        # plt.title("Stacking Ensemble Confusion Matrix")
        # plt.show()

        return self.stacker
