#!/usr/bin/env python
"""Script directo para XGBoost Baseline"""

import os, json, pickle, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report, auc
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
base_dir = Path.cwd()
prep_dir = base_dir.parent / "02-Preprocessamiento"

print("✅ Importaciones OK | XGBoost:", xgb.__version__)
print(f"📁 Base: {base_dir} | Prep: {prep_dir}")

# CARGAR DATOS
try:
    X_train = pd.read_csv(prep_dir / 'X_train_normalizado.csv')
    X_test = pd.read_csv(prep_dir / 'X_test_normalizado.csv')
    print(f"✅ Datos cargados: Train {X_train.shape}, Test {X_test.shape}")
except Exception as e:
    print(f"❌ Error cargando datos: {e}")
    sys.exit(1)

FEATURE_COLS = [c for c in X_train.columns if c != 'Propenso_a_Fallar']
y_train = X_train['Propenso_a_Fallar']
X_train = X_train[FEATURE_COLS]
y_test = X_test['Propenso_a_Fallar']
X_test = X_test[FEATURE_COLS]

print(f"Target distribution - Train: Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")

# CONFIG
splits = [{'nome': '60-40', 'test_size': 0.40}, {'nome': '70-30', 'test_size': 0.30}, {'nome': '80-20', 'test_size': 0.20}]
params = {'objective': 'binary:logistic', 'max_depth': 5, 'learning_rate': 0.05, 'n_estimators': 200, 'subsample': 0.8, 'random_state': 42, 'verbosity': 0}

results = {}
models = {}

# TRAIN
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

for cfg in splits:
    name = cfg['nome']
    print(f"\n>>> {name} Split")
    
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=cfg['test_size'], random_state=42, stratify=y_train)
    
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr)
    
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc_r = roc_auc_score(y_val, y_proba)
    
    print(f"  Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc_r:.4f}")
    
    results[name] = {'acc': acc, 'f1': f1, 'auc': auc_r, 'prec': precision_score(y_val, y_pred), 'rec': recall_score(y_val, y_pred), 
                     'y_val': y_val, 'y_pred': y_pred, 'y_proba': y_proba, 'cm': confusion_matrix(y_val, y_pred)}
    models[name] = model

best = max(results, key=lambda x: results[x]['auc'])
print(f"\n🏆 Best: {best} (AUC: {results[best]['auc']:.4f})")

# VISUALIZE
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# ROC
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (name, ax) in enumerate(zip(results.keys(), axes)):
    fpr, tpr, _ = roc_curve(results[name]['y_val'], results[name]['y_proba'])
    auc_s = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.5, label=f'AUC={auc_s:.4f}')
    ax.plot([0,1], [0,1], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('FPR', fontweight='bold')
    ax.set_ylabel('TPR', fontweight='bold')
    ax.set_title(f'ROC - {name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=100)
print("✅ roc_curve.png")
plt.close()

# CM
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for (name, ax) in zip(results.keys(), axes):
    sns.heatmap(results[name]['cm'], annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title(f'CM - {name}', fontweight='bold')
    ax.set_xticklabels(['No Fallo', 'Fallo'])
    ax.set_yticklabels(['No Fallo', 'Fallo'])
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=100)
print("✅ confusion_matrix.png")
plt.close()

# Feature Importance
best_model = models[best]
fi = pd.DataFrame({'Feature': FEATURE_COLS, 'Importance': best_model.feature_importances_}).sort_values('Importance', ascending=False)
fig, ax = plt.subplots(figsize=(12, 6))
top_fi = fi.head(15)
ax.barh(range(len(top_fi)), top_fi['Importance'].values, color=plt.cm.viridis(np.linspace(0, 1, len(top_fi))))
ax.set_yticks(range(len(top_fi)))
ax.set_yticklabels(top_fi['Feature'].values)
ax.set_xlabel('Importance', fontweight='bold')
ax.set_title(f'Top 15 Features - Split {best}', fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=100)
print("✅ feature_importance.png")
plt.close()

# REPORT
rep = f"""{'='*80}
XGBOOST BASELINE REPORT
{'='*80}
Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

RESULTS SUMMARY:
{'='*80}
"""
for name in results:
    rep += f"""\n{name} Split:
  Accuracy:  {results[name]['acc']:.4f}
  Precision: {results[name]['prec']:.4f}
  Recall:    {results[name]['rec']:.4f}
  F1-Score:  {results[name]['f1']:.4f}
  AUC-ROC:   {results[name]['auc']:.4f}
  CM: {results[name]['cm'].ravel()}
"""

rep += f"\n{'='*80}\nBEST MODEL: {best}\nAUC-ROC: {results[best]['auc']:.4f}\n{'='*80}\n"

with open('reporte_baseline.txt', 'w') as f:
    f.write(rep)
print("✅ reporte_baseline.txt")

# JSON
metricas = {'fecha': datetime.now().isoformat(), 'mejor_modelo': best, 'resultados': {}}
for name in results:
    metricas['resultados'][name] = {'auc': float(results[name]['auc']),  'f1': float(results[name]['f1']), 'acc': float(results[name]['acc']), 'prec': float(results[name]['prec']), 'rec': float(results[name]['rec'])}

with open('metricas_baseline.json', 'w') as f:
    json.dump(metricas, f, indent=2)
print("✅ metricas_baseline.json")

# SAVE MODEL
pickle.dump(best_model, open('modelo_baseline_xgboost.pkl', 'wb'))
print("✅ modelo_baseline_xgboost.pkl")

print("\n" + "="*80)
print(f"✅ COMPLETE! Best model: {best} (AUC: {results[best]['auc']:.4f})")
print("="*80)
