"""
Extract and display ML workflow results from trained models
Properly handles log-transformed features
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

print("="*80)
print("ML WORKFLOW RESULTS EXTRACTION")
print("="*80)

# Load trial results
print("\n[1] Loading trial data...")
df = pd.read_csv('in_silico_trial_results.csv')
print(f"Total simulations: {len(df)}")
print(f"Treatments: {df['treatment'].unique()}")
print(f"Immune statuses: {df['immune_status'].unique()}")

# Construct features with LOG-TRANSFORMATIONS
print("\n[2] Constructing features with log-transformations...")

# Create base features
features_df = pd.DataFrame()
features_df['age'] = df['age']
features_df['weight'] = df['weight']

# LOG-TRANSFORMED FEATURES (critical fix)
features_df['log_MIC'] = np.log10(df['MIC'])
features_df['log_initial_burden'] = np.log10(df['initial_burden'])

# One-hot encode immune_status
immune_dummies = pd.get_dummies(df['immune_status'], prefix='immune_status')
for col in ['immune_status_hyperinflammatory', 'immune_status_immunocompetent',
            'immune_status_immunosuppressed', 'immune_status_neutropenic']:
    features_df[col] = immune_dummies[col] if col in immune_dummies else 0

# One-hot encode infection_site
site_dummies = pd.get_dummies(df['infection_site'], prefix='infection_site')
for col in ['infection_site_bloodstream', 'infection_site_intra_abdominal',
            'infection_site_pneumonia', 'infection_site_skin_soft_tissue',
            'infection_site_urinary_tract']:
    features_df[col] = site_dummies[col] if col in site_dummies else 0

# One-hot encode treatment
treatment_dummies = pd.get_dummies(df['treatment'], prefix='treatment')
for col in treatment_dummies.columns:
    features_df[col] = treatment_dummies[col]

print(f"Features constructed: {features_df.shape[1]} features")
print(f"Feature names: {list(features_df.columns)}")

# Prepare targets
targets = {
    'clinical_success': df['clinical_success'],
    'microbiologic_success': df['microbiologic_success'],
    'resistance_emergence': df['resistance_emergence'],
    'inflammatory_toxicity': df['inflammatory_toxicity']
}

# Load trained models
print("\n[3] Loading trained models...")
with open('qsp_metamodel_nn.pkl', 'rb') as f:
    nn_model = pickle.load(f)

with open('qsp_metamodel_rf.pkl', 'rb') as f:
    rf_model = pickle.load(f)

print("Models loaded successfully")
print(f"Model type: {nn_model['model_type']}")
print(f"Expected features: {nn_model['feature_names']}")

# Train-test split (same random seed as training)
print("\n[4] Creating train-test split...")
# Split based on clinical_success for stratification
X_train, X_test, y_clin_train, y_clin_test = train_test_split(
    features_df,
    targets['clinical_success'].values,
    test_size=0.2,
    random_state=42,
    stratify=targets['clinical_success']
)

# Create dictionaries of all targets for train and test
y_train_dict = {}
y_test_dict = {}

train_idx = X_train.index
test_idx = X_test.index

for endpoint in ['clinical_success', 'microbiologic_success', 'resistance_emergence', 'inflammatory_toxicity']:
    y_train_dict[endpoint] = targets[endpoint].iloc[train_idx].values
    y_test_dict[endpoint] = targets[endpoint].iloc[test_idx].values

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Evaluate Neural Network
print("\n" + "="*80)
print("NEURAL NETWORK PERFORMANCE")
print("="*80)

nn_results = {}
for endpoint in ['clinical_success', 'microbiologic_success', 'resistance_emergence', 'inflammatory_toxicity']:
    print(f"\n--- {endpoint.replace('_', ' ').upper()} ---")

    # Get model and scaler for this endpoint
    model = nn_model['models'][endpoint]
    scaler = nn_model['scalers'][endpoint]

    # Scale test features
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    y_true = y_test_dict[endpoint]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Check if both classes are present for AUC calculation
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
    else:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"Accuracy: {accuracy:.3f}")
    if not np.isnan(auc):
        print(f"AUC: {auc:.3f}")
    else:
        print(f"AUC: N/A (only one class in test set)")

    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {cm[0,0]:6d}              {cm[0,1]:6d}")
    print(f"Actual Positive        {cm[1,0]:6d}              {cm[1,1]:6d}")

    # Calculate sensitivity and specificity
    if cm[1,1] + cm[1,0] > 0:
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    else:
        sensitivity = 0.0

    if cm[0,0] + cm[0,1] > 0:
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    else:
        specificity = 0.0

    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

    nn_results[endpoint] = {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

# Evaluate Random Forest
print("\n" + "="*80)
print("RANDOM FOREST PERFORMANCE")
print("="*80)

rf_results = {}
feature_importance = {}

for endpoint in ['clinical_success', 'microbiologic_success', 'resistance_emergence', 'inflammatory_toxicity']:
    print(f"\n--- {endpoint.replace('_', ' ').upper()} ---")

    # Get model and scaler for this endpoint
    model = rf_model['models'][endpoint]
    scaler = rf_model['scalers'][endpoint]

    # Scale test features
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_true = y_test_dict[endpoint]

    # Get probability for positive class (handle single-class case)
    proba_output = model.predict_proba(X_test_scaled)
    if proba_output.shape[1] > 1:
        y_pred_proba = proba_output[:, 1]
    else:
        # Only one class - use the single probability column
        y_pred_proba = proba_output[:, 0]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Check if both classes are present for AUC calculation
    unique_classes = np.unique(y_true)
    if len(unique_classes) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
    else:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    print(f"Accuracy: {accuracy:.3f}")
    if not np.isnan(auc):
        print(f"AUC: {auc:.3f}")
    else:
        print(f"AUC: N/A (only one class in test set)")

    print(f"\nConfusion Matrix:")
    print(f"                Predicted Negative  Predicted Positive")
    print(f"Actual Negative        {cm[0,0]:6d}              {cm[0,1]:6d}")
    print(f"Actual Positive        {cm[1,0]:6d}              {cm[1,1]:6d}")

    # Calculate sensitivity and specificity
    if cm[1,1] + cm[1,0] > 0:
        sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    else:
        sensitivity = 0.0

    if cm[0,0] + cm[0,1] > 0:
        specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    else:
        specificity = 0.0

    print(f"Sensitivity: {sensitivity:.3f}")
    print(f"Specificity: {specificity:.3f}")

    rf_results[endpoint] = {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity
    }

    # Feature importance
    importances = model.feature_importances_
    feature_importance[endpoint] = sorted(
        zip(nn_model['feature_names'], importances),
        key=lambda x: x[1],
        reverse=True
    )

# Summary comparison table
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Endpoint': [],
    'NN_Accuracy': [],
    'NN_AUC': [],
    'RF_Accuracy': [],
    'RF_AUC': [],
    'Winner_Accuracy': [],
    'Winner_AUC': []
})

for endpoint in ['clinical_success', 'microbiologic_success', 'resistance_emergence', 'inflammatory_toxicity']:
    nn = nn_results[endpoint]
    rf = rf_results[endpoint]

    winner_acc = 'NN' if nn['accuracy'] > rf['accuracy'] else 'RF' if rf['accuracy'] > nn['accuracy'] else 'Tie'

    # Handle NaN values for AUC comparison
    if np.isnan(nn['auc']) and np.isnan(rf['auc']):
        winner_auc = 'N/A'
    elif np.isnan(nn['auc']):
        winner_auc = 'RF'
    elif np.isnan(rf['auc']):
        winner_auc = 'NN'
    else:
        winner_auc = 'NN' if nn['auc'] > rf['auc'] else 'RF' if rf['auc'] > nn['auc'] else 'Tie'

    # Format AUC values, handling NaN
    nn_auc_str = f"{nn['auc']:.3f}" if not np.isnan(nn['auc']) else 'N/A'
    rf_auc_str = f"{rf['auc']:.3f}" if not np.isnan(rf['auc']) else 'N/A'

    comparison_df = pd.concat([comparison_df, pd.DataFrame({
        'Endpoint': [endpoint],
        'NN_Accuracy': [f"{nn['accuracy']:.3f}"],
        'NN_AUC': [nn_auc_str],
        'RF_Accuracy': [f"{rf['accuracy']:.3f}"],
        'RF_AUC': [rf_auc_str],
        'Winner_Accuracy': [winner_acc],
        'Winner_AUC': [winner_auc]
    })], ignore_index=True)

print("\n" + comparison_df.to_string(index=False))

# Feature importance for clinical_success
print("\n" + "="*80)
print("TOP 10 FEATURE IMPORTANCES (Random Forest - Clinical Success)")
print("="*80)

print("\nRank  Feature                               Importance")
print("-" * 60)
for i, (feature, importance) in enumerate(feature_importance['clinical_success'][:10], 1):
    print(f"{i:2d}.   {feature:40s}  {importance:.4f}")

# Overall summary statistics
print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)

# Calculate means, ignoring NaN values
nn_aucs = [v['auc'] for v in nn_results.values() if not np.isnan(v['auc'])]
rf_aucs = [v['auc'] for v in rf_results.values() if not np.isnan(v['auc'])]

print(f"\nNeural Network:")
print(f"  Mean Accuracy: {np.mean([v['accuracy'] for v in nn_results.values()]):.3f}")
if len(nn_aucs) > 0:
    print(f"  Mean AUC: {np.mean(nn_aucs):.3f} (computed from {len(nn_aucs)}/4 endpoints)")
else:
    print(f"  Mean AUC: N/A")

print(f"\nRandom Forest:")
print(f"  Mean Accuracy: {np.mean([v['accuracy'] for v in rf_results.values()]):.3f}")
if len(rf_aucs) > 0:
    print(f"  Mean AUC: {np.mean(rf_aucs):.3f} (computed from {len(rf_aucs)}/4 endpoints)")
else:
    print(f"  Mean AUC: N/A")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
