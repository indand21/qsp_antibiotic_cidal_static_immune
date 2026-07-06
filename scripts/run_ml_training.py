"""Run ML metamodel training and capture endpoint-specific metrics."""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

np.random.seed(42)

# Generate synthetic QSP simulation data (400 patients, 10 features, 4 endpoints)
n = 400
features = {
    'dose_mg': np.random.uniform(100, 2000, n),
    'interval_h': np.random.choice([4, 6, 8, 12, 24], n),
    'drug_class_cid': np.random.choice([0, 1], n),  # 0=static, 1=cidal
    'immune_strength': np.random.uniform(0.1, 1.5, n),
    'initial_burden': np.random.uniform(1e4, 1e7, n),
    'weight_kg': np.random.normal(70, 15, n),
    'ec50': np.random.uniform(0.3, 2.0, n),
    'e_max': np.random.uniform(3, 100, n),
    'hill_coeff': np.random.uniform(1.0, 2.5, n),
    'k_growth': np.random.uniform(0.3, 0.7, n),
}
X = pd.DataFrame(features)

# Generate endpoints based on features + noise
clinical_success = (
    (X['drug_class_cid'] * 0.4 +
     X['immune_strength'] * 0.3 +
     X['dose_mg']/2000 * 0.2 -
     X['interval_h']/24 * 0.1 +
     np.random.normal(0, 0.15, n)) > 0.4
).astype(int)

micro_success = (
    (X['drug_class_cid'] * 0.3 +
     X['immune_strength'] * 0.25 +
     X['dose_mg']/2000 * 0.25 -
     X['interval_h']/24 * 0.1 +
     np.random.normal(0, 0.15, n)) > 0.35
).astype(int)

resistance = (
    (X['interval_h']/24 * 0.4 +
     X['initial_burden']/1e7 * 0.2 +
     X['drug_class_cid'] * -0.15 +
     np.random.normal(0, 0.15, n)) > 0.3
).astype(int)

toxicity = (
    (X['immune_strength'] * 0.5 +
     X['initial_burden']/1e7 * 0.2 +
     X['drug_class_cid'] * 0.15 +
     np.random.normal(0, 0.15, n)) > 0.35
).astype(int)

endpoints = {
    'clinical_success': clinical_success,
    'microbiological_success': micro_success,
    'resistance_emergence': resistance,
    'inflammatory_toxicity': toxicity,
}

results = {}

for name, y in endpoints.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # NN
    nn = MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42)
    nn.fit(X_train_s, y_train)
    nn_pred = nn.predict_proba(X_test_s)[:, 1]
    nn_auc = roc_auc_score(y_test, nn_pred) if len(np.unique(y_test)) > 1 else 0.5
    nn_acc = accuracy_score(y_test, nn.predict(X_test_s))

    # RF
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict_proba(X_test_s)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_pred) if len(np.unique(y_test)) > 1 else 0.5
    rf_acc = accuracy_score(y_test, rf.predict(X_test_s))

    results[name] = {
        'nn_auc': round(nn_auc, 3),
        'nn_accuracy': round(nn_acc, 3),
        'rf_auc': round(rf_auc, 3),
        'rf_accuracy': round(rf_acc, 3),
        'n_positive': int(y.sum()),
        'n_negative': int(len(y) - y.sum()),
    }
    print(f"{name:30s} NN AUC={nn_auc:.3f} Acc={nn_acc:.3f} | RF AUC={rf_auc:.3f} Acc={rf_acc:.3f} | N+={int(y.sum())}/{n}")

print("\n=== Summary ===")
for name, r in results.items():
    print(f"  {name}: NN AUC={r['nn_auc']:.3f}, NN Acc={r['nn_accuracy']:.3f}, RF AUC={r['rf_auc']:.3f}, RF Acc={r['rf_accuracy']:.3f}")

mean_nn_auc = np.mean([r['nn_auc'] for r in results.values()])
mean_rf_auc = np.mean([r['rf_auc'] for r in results.values()])
mean_nn_acc = np.mean([r['nn_accuracy'] for r in results.values()])
mean_rf_acc = np.mean([r['rf_accuracy'] for r in results.values()])
print(f"\nMean NN AUC: {mean_nn_auc:.3f}")
print(f"Mean RF AUC: {mean_rf_auc:.3f}")
print(f"Mean NN Acc: {mean_nn_acc:.3f}")
print(f"Mean RF Acc: {mean_rf_acc:.3f}")
