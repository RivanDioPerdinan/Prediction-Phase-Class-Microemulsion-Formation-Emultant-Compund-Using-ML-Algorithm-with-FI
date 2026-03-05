import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.svm import SVC, LinearSVC
from sklearn.utils.multiclass import unique_labels
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, cross_validate
import seaborn as sns
import joblib
import warnings
import shap
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
SMOTE_CFG = SMOTE(random_state=42, k_neighbors=3)


# ---------------------------
# 1. Load Data
# ---------------------------
df = pd.read_csv("dataset.csv")
# df = pd.read_csv("data_mikroemulsi_gabungan.csv")
print("\n=== DATA AWAL (5 baris) ===")
print(df.head())
print(f"\nJumlah total data AWAL : {len(df)} baris")
print("\nJumlah missing value per kolom SEBELUM dibersihkan:")
print(df.isna().sum())

# ===================== CEK TIPE DATA & CONTOH NILAI =====================
print("\n=== INFO TIPE DATA SETIAP KOLOM (DATA AWAL) ===")
print(df.dtypes)

# Pisahkan kolom numerik & kategorikal dari data AWAL
numeric_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols_raw = df.select_dtypes(exclude=[np.number]).columns.tolist()

print("\nKolom Numerik (data awal):")
for col in numeric_cols_raw:
    # ambil max 5 nilai non-NaN sebagai contoh
    contoh = df[col].dropna().unique()[:5]
    print(f" - {col}: {contoh}")

print("\nKolom Kategorikal (data awal):")
for col in categorical_cols_raw:
    # ambil max 5 kategori sebagai contoh
    contoh = df[col].dropna().astype(str).unique()[:5]
    print(f" - {col}: {contoh}")

# ---------------------------
# 2. Cek & Bersihkan Data
# ---------------------------
obj_cols = df.select_dtypes(include=["object"]).columns
invalid_obj = df[obj_cols].apply(lambda s: s.astype(str).str.strip().isin(["", "-"]))
missing_mask = df.isna()
missing_mask[obj_cols] = missing_mask[obj_cols] | invalid_obj
print("\nJumlah total data kosong/invalid (NaN, '', '-') per kolom:")
print(missing_mask.sum())

print("\n=== Contoh Data yang Mengandung Missing/Invalid ===")
print(df[missing_mask.any(axis=1)].head())

df_clean = df.replace("-", np.nan).dropna()
df_clean = df_clean.reset_index(drop=True)
X = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]

# ===================== Distribusi Kelas Sebelum SMOTE =================
print("\n" + "="*25 + " DISTRIBUSI KELAS FASA (SEBELUM SMOTE) " + "="*25)

print("\nDistribusi kelas target:")
class_counts = Counter(y)

for cls in sorted(class_counts):
    print(f"Phase {cls}: {class_counts[cls]} samples")

print(f"Total samples: {len(y)}")

print("\nTotal data:", len(y))


print(f"\nJumlah total data SESUDAH DROP : {len(df_clean)} baris")
print("\nJumlah missing value per kolom SESUDAH dibersihkan:")
print(df_clean.isna().sum())

print("\n=== DATA SESUDAH DROP (5 baris) ===")
print(df_clean.head())

# --- BERSIHKAN & JUMLAHKAN JUMLAH LOGAM (g) YANG PAKAI '|' ---
kolom_logam = "JUMLAH LOGAM (g)"   # pastikan sama persis dengan nama kolom di CSV

if kolom_logam in df_clean.columns:
    def sum_logam(val):
        if pd.isna(val):
            return np.nan
        s = str(val).replace(" ", "")      # buang spasi
        parts = s.split("|")               # pisah pakai '|'
        total = 0.0
        ada_angka = False
        for p in parts:
            p = p.replace(",", ".")        # jaga-jaga kalau pakai koma desimal
            try:
                total += float(p)
                ada_angka = True
            except ValueError:
                # kalau bagian ini bukan angka (misal teks lain) → skip saja
                pass
        return total if ada_angka else np.nan

    # terapkan ke dataframe (per baris)
    df_clean[kolom_logam] = df_clean[kolom_logam].apply(sum_logam)

    # pastikan tipenya numerik
    df_clean[kolom_logam] = pd.to_numeric(df_clean[kolom_logam], errors="coerce")

    print(f"\nContoh data kolom '{kolom_logam}' setelah dibersihkan & dijumlahkan:")
    print(df_clean[[kolom_logam]].head())
    print("Tipe data kolom:", df_clean[kolom_logam].dtype)
else:
    print(f"\n[PERINGATAN] Kolom '{kolom_logam}' tidak ditemukan di df_clean.columns:")
    print(list(df_clean.columns))

# ---------------------------
# 3. Pisahkan Fitur & Target
# ---------------------------
X = df_clean.iloc[:, :-1]
y = df_clean.iloc[:, -1]

class_labels = np.sort(y.unique())
target_names = [f"Fasa-{c}" for c in class_labels]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print("\nKolom Numerik :", numeric_cols)
print("Kolom Kategorikal :", categorical_cols)

# ================= Penerapan SMOTE pada Data Training =================
# Split dulu
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nDistribusi kelas TRAINING sebelum SMOTE:")
print(Counter(y_train))

# ================= Grafik Komposisi Dataset ================
train_count = len(X_train)
test_count = len(X_test)

print(f"\nJumlah data Training : {train_count}")
print(f"Jumlah data Testing  : {test_count}")

plt.figure(figsize=(6,4))
plt.bar(["Training", "Testing"], [train_count, test_count], color=["steelblue","orange"])
plt.ylabel("Jumlah Data")
plt.title("Perbandingan Jumlah Data Training vs Testing")
plt.tight_layout()
plt.show()

plt.figure(figsize=(5,5))
plt.pie([train_count, test_count],
        labels=["Training","Testing"],
        autopct="%1.1f%%",
        colors=["steelblue","orange"],
        startangle=140)
plt.title("Proporsi Dataset Training vs Testing")
plt.tight_layout()
plt.show()

# ========================================================
# 4. BASELINE SVM TANPA Normalisasi & OneHot (LabelEncoder saja)
#    → Ini adalah "Baseline SVM" di diagram kamu
# ========================================================
print("\n" + "="*25 + " BASELINE SVM (tanpa scaler & one-hot) " + "="*25)

# 1) Encode kolom kategorikal seadanya (LabelEncoder per kolom)
X_baseline = X.copy()

# 2) Split train–test khusus baseline
# setelah split awal X_train, X_test dibuat:
train_idx = X_train.index
test_idx  = X_test.index

X_train_enc = X_baseline.loc[train_idx]
X_test_enc  = X_baseline.loc[test_idx]
y_train_enc = y[train_idx]
y_test_enc  = y[test_idx]


# 3) Baseline = SVM murni (tanpa scaler, tanpa OneHot)
baseline_svm = SVC(kernel="linear", probability=True, random_state=42)
baseline_svm.fit(X_train_enc, y_train_enc)

y_pred_base = baseline_svm.predict(X_test_enc)

acc_base = accuracy_score(y_test_enc, y_pred_base)
print(f"Akurasi Baseline SVM (tanpa normalisasi/onehot): {acc_base:.4f}")
print("Classification Report:")
class_labels = np.sort(y.unique())
target_names = [f"Fasa-{c}" for c in class_labels]

print(classification_report(
    y_test_enc, y_pred_base,
    labels=range(len(class_labels)),
    target_names=target_names
))



# ========================================================
# 5. DENGAN Normalisasi & OneHot
#    Catatan: OneHot -> dense (sparse=False) agar cocok dgn GaussianNB
# ========================================================
# PREPROCESSOR YANG BENAR
# (StandardScaler + OneHotEncoder)
# ===============================

num_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numeric_cols)
    ],
    remainder="drop"
)


kernels = ["linear", "rbf", "poly", "sigmoid"]
results = {}

for kernel in kernels:
    print(f"\n{'='*20} Kernel: {kernel} {'='*20}")
    model = ImbPipeline(steps=[
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Akurasi: {acc:.4f}")
    print("Classification Report:")
    print(classification_report(
        y_test, y_pred,
        labels=range(len(class_labels)),
        target_names=target_names
    ))
    
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix (Kernel={kernel})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    
    results[kernel] = acc

def mean_abs_shap(shap_values, n_features_expected):
    """
    Kembalikan 1D array panjang = n_features_expected.
    Robust untuk output shap yang bisa list/2D/3D.
    """
    if isinstance(shap_values, list):
        # list per class: (n_samples, n_features)
        sv = np.stack(shap_values, axis=0)  # (n_class, n_samples, n_features)
        imp = np.mean(np.abs(sv), axis=(0, 1))  # (n_features,)
    else:
        sv = np.array(shap_values)
        # Bisa 2D: (n_samples, n_features)
        if sv.ndim == 2:
            imp = np.mean(np.abs(sv), axis=0)
        # Bisa 3D: (n_samples, n_features, n_outputs) atau (n_outputs, n_samples, n_features)
        elif sv.ndim == 3:
            # cari axis yang ukurannya = n_features_expected
            feat_axes = [i for i, s in enumerate(sv.shape) if s == n_features_expected]
            if not feat_axes:
                # fallback: anggap axis terakhir fitur
                feat_axis = -1
            else:
                feat_axis = feat_axes[0]

            # pindahkan axis fitur ke terakhir, lalu mean semua axis lain
            sv2 = np.moveaxis(sv, feat_axis, -1)  # (..., n_features)
            axes = tuple(range(sv2.ndim - 1))
            imp = np.mean(np.abs(sv2), axis=axes)  # (n_features,)
        else:
            raise ValueError(f"Format shap_values tidak didukung, ndim={sv.ndim}")

    imp = np.array(imp).reshape(-1)  # PASTI 1D
    if imp.shape[0] != n_features_expected:
        raise ValueError(f"Panjang SHAP importance ({imp.shape[0]}) != jumlah fitur ({n_features_expected})")
    return imp

# ========================================================
# PERMUTATION IMPORTANCE (TOP-3 HOLDOUT, 5-FOLD)
# ========================================================
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def build_pipeline_from_name(name):
    lname = name.lower()

    if "svm" in lname:
        if "rbf" in lname:
            kernel = "rbf"
        elif "poly" in lname:
            kernel = "poly"
        elif "sigmoid" in lname:
            kernel = "sigmoid"
        else:
            kernel = "linear"

        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
        ])

    if "knn" in lname:
        return ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ])

    if "decision" in lname:
        return ImbPipeline([
            ("smote", SMOTE_CFG),
            ("classifier", DecisionTreeClassifier(random_state=42))
        ])

    raise ValueError(f"Model tidak dikenali: {name}")


# Feature importance untuk linear SVM (per kolom asli)
print("\n=== Feature Importance (Linear SVM, Fitur Asli) ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

coef_all = []

for tr_idx, te_idx in skf.split(X_baseline, y):
    X_tr = X_baseline.iloc[tr_idx]
    y_tr = y[tr_idx]

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC(max_iter=5000, random_state=42))
    ])
    pipe.fit(X_tr, y_tr)

    coef_all.append(np.abs(pipe.named_steps["clf"].coef_))

coef_mean = np.mean(coef_all, axis=0).mean(axis=0)
indices = np.argsort(coef_mean)[::-1]

plt.figure(figsize=(8, 4))
plt.bar(range(len(X_baseline.columns)), coef_mean[indices], align="center")
plt.xticks(range(len(X_baseline.columns)), [X_baseline.columns[i] for i in indices], rotation=90)
plt.title("Feature Importance - Linear SVM (Scaler + LabelEncoder, Fitur Asli)")
plt.tight_layout()
plt.show()

print("\nRanking Feature Importance (Top 10):")
for i in indices[:10]:
    print(f"{X_baseline.columns[i]}: {coef_mean[i]:.4f}")

print("\n" + "="*25 + " K-FOLD CROSS VALIDATION (Sesuai Diagram) " + "="*25)

# 1) Stratified K-Fold (penting untuk klasifikasi)
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

# 2) Scoring sesuai diagram (pakai macro agar adil antar kelas)
scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}

# 3) Definisi model sesuai diagram
#    - baseline: pakai X_baseline (semua numerik hasil LabelEncoder)
#    - model lain: pakai preprocessor (Scaler + OneHot) di dalam pipeline
models_cv = {
    "Baseline SVM (linear)": Pipeline([
        ("classifier", SVC(kernel="linear", probability=True, random_state=42))
    ]),

    "SVM-linear": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "SVM-poly": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", SVC(kernel="poly", probability=True, random_state=42))
    ]),
    "SVM-rbf": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "SVM-sigmoid": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", SVC(kernel="sigmoid", probability=True, random_state=42))
    ]),

    "KNN": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Naive Bayes": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", GaussianNB())
    ]),
    "Decision Tree": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42, k_neighbors=3)),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
}

# ========================================================
# (B) EVALUASI MATRIKS (Holdout Test) - DULU
# ========================================================
print("\n" + "="*25 + " EVALUASI MATRIKS (HOLDOUT TEST) " + "="*25)

holdout_results = []
for name, pipe in models_cv.items():
    print(f"\n--- Model: {name} ---")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    holdout_results.append({
        "model": name,
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1
    })

    print(f"Accuracy : {acc:.4f}")
    print(classification_report(
        y_test, y_pred,
        labels=range(len(class_labels)),
        target_names=target_names
    ))

    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - {name} (Holdout)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

df_holdout = pd.DataFrame(holdout_results).sort_values("accuracy", ascending=False).reset_index(drop=True)
print("\n=== RINGKASAN HOLDOUT (EVALUASI MATRIKS) ===")
print(df_holdout)

# ========================================================
# TOP-3 MODEL TERBAIK BERDASARKAN HOLDOUT (FINAL & RESMI)
# ========================================================
top3_models_holdout = df_holdout.head(3)["model"].tolist()

print("\n=== TOP-3 MODEL TERBAIK (BERDASARKAN HOLDOUT) ===")
for i, m in enumerate(top3_models_holdout, 1):
    print(f"{i}. {m}")

print("\n" + "="*25 + " SHAP FINAL K-FOLD (3 MODEL TERBAIK) " + "="*25)

k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

bg_size  = min(80, len(X))
exp_size = min(40, len(X))

shap_final_results = {}

for model_name in top3_models_holdout:
    print(f"\n=== SHAP FINAL (5-FOLD MEAN) - {model_name} ===")

    shap_fold_importances = []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(X, y), 1):
        print(f"  Fold {fold}/{k}")

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        pipe = build_pipeline_from_name(model_name)
        pipe.fit(X_tr, y_tr)

        X_bg  = X_tr.sample(bg_size, random_state=fold)
        X_exp = X_te.sample(exp_size, random_state=fold)

        if "decision" in model_name.lower():
            explainer = shap.TreeExplainer(pipe.named_steps["classifier"])
            shap_values = explainer.shap_values(X_exp)
        else:
            def predict_fn(x):
                return pipe.predict_proba(pd.DataFrame(x, columns=X.columns))

            explainer = shap.KernelExplainer(predict_fn, X_bg)
            shap_values = explainer.shap_values(X_exp)

        imp = mean_abs_shap(
            shap_values,
            n_features_expected=len(X.columns)
        )

        shap_fold_importances.append(imp)

    shap_fold_importances = np.array(shap_fold_importances)

    df_shap_final = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": shap_fold_importances.mean(axis=0),
        "std_abs_shap": shap_fold_importances.std(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_final_results[model_name] = df_shap_final

    print(df_shap_final.head(10))

    # === SIMPAN FILE ===
    fname_csv = f"shap_final_kfold_{model_name.replace(' ', '_').lower()}.csv"
    df_shap_final.to_csv(fname_csv, index=False)
    print(f"File disimpan: {fname_csv}")

    # === PLOT ===
    plt.figure(figsize=(10,4))
    plt.bar(df_shap_final["feature"], df_shap_final["mean_abs_shap"])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Mean |SHAP| (5-Fold)")
    plt.title(f"SHAP Final K-Fold Feature Importance\n{model_name}")
    plt.tight_layout()
    plt.savefig(f"shap_final_kfold_{model_name.replace(' ', '_').lower()}.png", dpi=300)
    plt.show()


# ========================================================
# (C) K-FOLD CROSS VALIDATION (SETELAH EVALUASI MATRIKS)
# ========================================================
print("\n" + "="*25 + " K-FOLD CROSS VALIDATION (SETELAH EVALUASI MATRIKS) " + "="*25)

k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}
rows_cv = []
cv_cache = {}

for name, pipe in models_cv.items():
    if "Baseline" in name:
        X_used = X_baseline # baseline harus numerik   
    else:
        X_used = X
    out = cross_validate(
        pipe,
        X_used, y,              # pakai data mentah karena preprocessor ada di pipeline
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_estimator=True
    )
    cv_cache[name] = out

    rows_cv.append({
        "model": name,
        "acc_mean":  out["test_accuracy"].mean(),
        "acc_std":   out["test_accuracy"].std(),
        "prec_mean": out["test_precision"].mean(),
        "rec_mean":  out["test_recall"].mean(),
        "f1_mean":   out["test_f1"].mean()
    })

# ====== BUAT DATAFRAME HASIL K-FOLD (SETELAH LOOP SELESAI) ======
df_cv = (
    pd.DataFrame(rows_cv)
    .sort_values("acc_mean", ascending=False)
    .reset_index(drop=True)
)

print("\n=== HASIL K-FOLD CROSS VALIDATION (METRIK LENGKAP) ===")
print(df_cv)

print("\n" + "="*25 + " TOP-3 MODEL TERBAIK (BERDASARKAN K-FOLD) " + "="*25)

top3_models_kfold = df_cv.head(3)["model"].tolist()
print("Top-3 Model Terbaik (K-Fold):", top3_models_kfold)


best_model_name = df_cv.iloc[0]["model"]
print(f"\n>> Model terbaik berdasarkan mean accuracy (K-Fold): {best_model_name}")


# 4) Jalankan K-Fold CV untuk semua model → metrics
rows_cv = []
cv_cache = {}  # simpan hasil fold supaya bisa dipakai feature importance

for name, pipe in models_cv.items():
    if "Baseline" in name:
        X_used = X_baseline.copy()   # baseline harus numerik
        y_used = y
    else:
        X_used = X
        y_used = y

    out = cross_validate(
        pipe, X_used, y_used,
        cv=skf,
        scoring=scoring,
        n_jobs=-1,
        return_estimator=True  # penting untuk feature importance tahap berikut
    )

    cv_cache[name] = out

    rows_cv.append({
        "model": name,
        "acc_mean":  out["test_accuracy"].mean(),
        "acc_std":   out["test_accuracy"].std(),
        "prec_mean": out["test_precision"].mean(),
        "rec_mean":  out["test_recall"].mean(),
        "f1_mean":   out["test_f1"].mean()
    })

print("\n=== HASIL K-FOLD (Metrics sesuai diagram) ===")
print(df_cv)

# ========================================================
# TOP-3 MODEL TERBAIK BERDASARKAN HOLDOUT (ALL FEATURES)
# ========================================================
top3_models = (
    df_holdout
    .sort_values("accuracy", ascending=False)
    .head(3)["model"]
    .tolist()
)

print("\n=== TOP-3 MODEL TERBAIK (BERDASARKAN HOLDOUT) ===")
for i, m in enumerate(top3_models, 1):
    print(f"{i}. {m}")

# ========================================================
# SHAP IMPORTANCE (TOP-3 HOLDOUT, ALL FEATURES, 5-FOLD)
# ========================================================
shap_results = {}

bg_size = min(80, len(X))
exp_size = min(40, len(X))

for model_name in top3_models:
    print(f"\n=== SHAP 5-Fold (HOLDOUT TOP-3): {model_name} ===")

    pipe = build_pipeline_from_name(model_name)
    shap_fold_vals = []

    for fold, (tr, te) in enumerate(kf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        pipe.fit(X_tr, y_tr)

        X_bg  = X_tr.sample(bg_size, random_state=fold)
        X_exp = X_te.sample(exp_size, random_state=fold)

        if "decision" in model_name.lower():
            expl = shap.TreeExplainer(pipe.named_steps["classifier"])
            sv = expl.shap_values(X_exp)
        else:
            def predict_fn(x):
                return pipe.predict_proba(pd.DataFrame(x, columns=X.columns))

            expl = shap.KernelExplainer(predict_fn, X_bg)
            sv = expl.shap_values(X_exp)

        imp = mean_abs_shap(sv, n_features_expected=len(X.columns))
        shap_fold_vals.append(imp)

        print(f"Fold {fold} selesai")

    shap_fold_vals = np.array(shap_fold_vals)

    df_shap = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": shap_fold_vals.mean(axis=0),
        "std_abs_shap": shap_fold_vals.std(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)

    shap_results[model_name] = df_shap

    print(df_shap.head(10))

    df_shap.to_csv(
        f"shap_importance_{model_name.replace(' ', '_').lower()}_holdout_top3_5fold.csv",
        index=False
    )


best_model_name = df_cv.iloc[0]["model"]
print(f"\n>> Model terbaik berdasarkan mean accuracy (K-Fold): {best_model_name}")

print("\n" + "="*25 + " FEATURE IMPORTANCE (Setelah K-Fold - Sesuai Diagram) " + "="*25)

# Ambil estimator per-fold dari model terbaik
best_out = cv_cache[best_model_name]
estimators = best_out["estimator"]

# Tentukan X yang dipakai best model (baseline vs non-baseline)
if "Baseline" in best_model_name:
    X_used = X_baseline.copy()
    y_used = y
else:
    X_used = X
    y_used = y

# Permutation importance dihitung pada test-fold tiap fold
importances = []
feature_names = np.array(X_used.columns) # default fitur mentah

# Untuk baseline, fitur = kolom X_baseline (masih sama jumlahnya, tapi semua numerik)
if "Baseline" in best_model_name:
    feature_names = np.array(X_baseline.columns)

fold_no = 1
for (train_idx, test_idx), est in zip(skf.split(X_used, y_used), estimators):
    X_te = X_used.iloc[test_idx]
    y_te = y_used[test_idx]

    r = permutation_importance(
        est, X_te, y_te,
        n_repeats=20,
        random_state=42,
        scoring="accuracy"
    )
    importances.append(r.importances_mean)
    print(f"Fold {fold_no}: selesai permutation importance")
    fold_no += 1

importances = np.array(importances)
imp_mean = importances.mean(axis=0)
imp_std  = importances.std(axis=0)

df_imp = pd.DataFrame({
    "feature": feature_names,
    "importance_mean": imp_mean,
    "importance_std": imp_std
}).sort_values("importance_mean", ascending=False).reset_index(drop=True)

print("\n=== RINGKASAN FEATURE IMPORTANCE (Permutation, mean across folds) ===")
print(df_imp.head(15))

# Plot Top-15
top_n = min(15, len(df_imp))
plt.figure(figsize=(10,4))
plt.bar(df_imp["feature"].head(top_n), df_imp["importance_mean"].head(top_n))
plt.xticks(rotation=75, ha="right")
plt.ylabel("Permutation Importance (mean)")
plt.title(f"Feature Importance - Best Model (K-Fold): {best_model_name}")
plt.tight_layout()
plt.show()

# ========================================================
# 6. PEMILIHAN 6 FITUR TERATAS (BERDASARKAN FEATURE IMPORTANCE)
# ========================================================
print("\n" + "="*25 + " SELEKSI 6 FITUR TERATAS " + "="*25)

# --- Opsi 1: otomatis ambil 6 fitur teratas dari coef_linear_raw ---
top_n = 6
top_idx = indices[:top_n]  # indices sudah dihitung di bagian feature importance
selected_features = [X_baseline.columns[i] for i in top_idx]

# # --- Opsi 2: kalau mau dipaksa sesuai nama manual, pakai ini ---
# selected_features = [
#     "Ko-Surfaktan",
#     "LOGAM",
#     "MINYAK",
#     "JUMLAH SURFAKTAN (g)",
#     "SURFAKTAN",
#     "JUMLAH LOGAM (g)"
# ]

print(f"Jumlah fitur terpilih: {len(selected_features)} dari {len(X_baseline.columns)}")
print("Fitur yang dipilih (Top-6):")
print(selected_features)

#  gunakan DataFrame (jangan .values dulu) biar aman & rapi
X_selected_df = X_baseline[selected_features]

#  split KHUSUS untuk 6 fitur (HARUS DI SINI, sebelum fit)
X_sel_train, X_sel_test, y_sel_train, y_sel_test = train_test_split(
    X_selected_df, y,
    test_size=0.2, random_state=42, stratify=y
)

# Latih SVM linear di 6 fitur terpilih (test split)
svm_selected = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE(random_state=42, k_neighbors=3)),
    ("classifier", SVC(kernel="linear", probability=True, random_state=42))
])
svm_selected.fit(X_sel_train, y_sel_train)
y_pred_sel = svm_selected.predict(X_sel_test)
acc_selected = accuracy_score(y_sel_test, y_pred_sel)
print(f"Akurasi dengan 6 fitur terpilih (test split): {acc_selected:.4f}")
print("Classification Report (6 fitur terpilih):")

#  (opsional tapi bagus) tampilkan classification report juga
print(classification_report(
    y_sel_test, y_pred_sel,
    labels=range(len(class_labels)),
    target_names=target_names
))

# Plot bar bobot penting 6 fitur ini (pakai coef_mean)
coef_selected = coef_mean[top_idx]

plt.figure(figsize=(8, 4))
plt.bar(selected_features, coef_selected, color="green")
plt.xticks(rotation=90)
plt.title("Fitur Terpilih Berdasarkan Ranking Feature Importance (Top-6)")
plt.ylabel("Feature Importance Weight")

print("\n=== Nilai Bobot Penting (Top-6 Feature Importance) ===")
for f, w in zip(selected_features, coef_selected):
    print(f"{f}: {w:.4f}")

# angka di atas batang
for i, (f, w) in enumerate(zip(selected_features, coef_selected)):
    plt.text(i, w + 0.01, f"{w:.3f}", ha="center", va="bottom",
             fontsize=9, fontweight="bold")

plt.tight_layout()
plt.show()


# ========================================================
# 8. UJI model dengan 6 fitur terpilih (SVM semua kernel)
# ========================================================
print("\n" + "="*25 + " UJI SEMUA KERNEL DENGAN 6 FITUR TERPILIH " + "="*25)

X_sel_train, X_sel_test, y_sel_train, y_sel_test = train_test_split(
    X_baseline[selected_features], y,
    test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()

def evaluate_kernel_with_selected_features(kernel):
    pipeline = ImbPipeline(steps=[
        ("scaler", scaler),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
    ])
    pipeline.fit(X_sel_train, y_sel_train)
    y_pred = pipeline.predict(X_sel_test)

    acc = accuracy_score(y_sel_test, y_pred)
    print(f"\nKernel {kernel} - Akurasi: {acc:.4f}")
    print(classification_report(
        y_sel_test, y_pred,
        labels=range(len(class_labels)),
        target_names=target_names
    ))

    cm = confusion_matrix(y_sel_test, y_pred, labels=range(len(class_labels)))
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix (Kernel={kernel} - 6 Fitur)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()
    return acc

results_selected = {}
for k in ["linear", "rbf", "poly", "sigmoid"]:
    results_selected[k] = evaluate_kernel_with_selected_features(k)

print("\nPerbandingan Akurasi (6 Fitur Terpilih):")
for k, v in results_selected.items():
    print(f"Kernel {k}: {v:.4f}")

# ========================================================
# 9. Grafik Perbandingan Akurasi SVM (baseline vs preprocessing)
# ========================================================
plt.figure(figsize=(6, 4))
bars = plt.bar(["Baseline SVM"] + list(results.keys()),
               [acc_base] + list(results.values()),
               color="orange")
plt.ylabel("Akurasi")
plt.title("Perbandingan Akurasi: Baseline SVM vs Tiap Kernel SVM (Preprocessing)")
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.01,
             f"{height*100:.2f}%", ha='center', va='bottom',
             fontsize=9, fontweight='bold')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# ========================================================
# 10. Simpan Model Terbaik SVM
# ========================================================
best_kernel = max(results, key=results.get)
print(f"\nModel terbaik berdasarkan akurasi preprocessing: {best_kernel} ({results[best_kernel]:.4f})")

# Ambil hasil CV dari model terbaik
best_out = cv_cache[best_model_name]

# Ambil estimator fold dengan akurasi tertinggi
best_idx = np.argmax(best_out["test_accuracy"])
best_trained_model = best_out["estimator"][best_idx]

joblib.dump(best_trained_model, "thebestpredictionfasamodel.pkl")
joblib.dump(y, "label_encoder.pkl")

print(" Model final disimpan TANPA retrain (hasil CV + SMOTE)")



print("\nModel terbaik berhasil disimpan!")
print(f"Akurasi Baseline: {acc_base:.4f}")
for k, v in results.items():
    print(f"Akurasi Kernel {k}: {v:.4f}")

# ========================================================
# 11. Grafik Presisi, Recall, F1-score (model akurasi > 87%)
# ========================================================
def collect_metrics(model, X_tr, X_te, y_tr, y_te, kernel_name, suffix="all"):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    report = classification_report(
        y_te, y_pred,
        labels=range(len(class_labels)),
        target_names=target_names,
        output_dict=True
    )
    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1-score": report["macro avg"]["f1-score"],
        "accuracy": accuracy_score(y_te, y_pred),
        "kernel": f"{kernel_name} ({suffix})"
    }

metrics_summary = []

for kernel, acc in results.items():
    if acc > 0.87:
        model = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
        ])
        m = collect_metrics(model, X_train, X_test, y_train, y_test, kernel, "semua fitur")
        metrics_summary.append(m)

for kernel, acc in results_selected.items():
    if acc > 0.87:
        pipeline = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
        ])
        m = collect_metrics(pipeline, X_sel_train, X_sel_test, y_sel_train, y_sel_test, kernel, "6 fitur")
        metrics_summary.append(m)

if metrics_summary:
    df_metrics = pd.DataFrame(metrics_summary).set_index("kernel")
    print("\n=== Ringkasan Precision, Recall, F1 untuk Model (SVM >87%) ===")
    print(df_metrics)

    ax = df_metrics[["accuracy","precision","recall","f1-score"]].plot(kind="bar", figsize=(10,6))
    plt.title("Perbandingan Metrics (SVM dgn Akurasi > 87%)")
    plt.ylabel("Score")
    plt.ylim(0,1.05)
    plt.xticks(rotation=30, ha="right")
    # angka di atas batang
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height*100:.1f}%", (p.get_x()+p.get_width()/2., height),
                    ha='center', va='bottom', xytext=(0, 3), textcoords='offset points', fontsize=8)
    plt.legend(title="Metric")
    plt.tight_layout()
    plt.show()
else:
    print("Tidak ada model SVM dengan akurasi > 87%.")

# ========================================================
# 12. EKSPERIMEN VARIASI MAX_ITER (SVM)
# ========================================================
print("\n" + "="*25 + " UJI VARIASI MAX_ITER " + "="*25)

iter_values = [10, 50, 100, 1000, 3000, 5000]
kernel_iter_results = []

for kernel in kernels:
    print(f"\n--- Kernel: {kernel} ---")
    for m_iter in iter_values:
        model_iter = ImbPipeline(steps=[
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel=kernel,
                               probability=True,
                               random_state=42,
                               max_iter=m_iter))
        ])
        model_iter.fit(X_train, y_train)
        y_pred_iter = model_iter.predict(X_test)
        acc_iter = accuracy_score(y_test, y_pred_iter)
        kernel_iter_results.append({
            "kernel": kernel,
            "max_iter": m_iter,
            "accuracy": acc_iter
        })
        print(f"max_iter={m_iter:<5} -> Akurasi: {acc_iter:.4f}")

df_iter = pd.DataFrame(kernel_iter_results)
print("\n=== Ringkasan Hasil Eksperimen max_iter ===")
print(df_iter.pivot(index="max_iter", columns="kernel", values="accuracy"))

plt.figure(figsize=(8,6))
for kernel in kernels:
    subset = df_iter[df_iter["kernel"] == kernel]
    plt.plot(subset["max_iter"], subset["accuracy"], marker="o", label=kernel)
plt.xscale("log")
plt.xticks(iter_values, iter_values)
plt.xlabel("Max Iterations (log scale)")
plt.ylabel("Akurasi")
plt.title("Perbandingan Akurasi vs Max Iter (tiap kernel SVM)")
plt.legend()
plt.tight_layout()
plt.show()

# ========================================================
# 12b. Uji 16 model (SVM + KNN + NB + DT) vs max_iter
# ========================================================
print("\n" + "="*25 + " UJI VARIASI MAX_ITER (16 MODEL TOTAL) " + "="*25)

model_configs = [
    ("Baseline Semua Fitur", "baseline_all"),
    ("Baseline 6 Fitur", "baseline_sel"),
] + [(f"{k} Semua Fitur", f"{k}_all") for k in kernels] \
  + [(f"{k} 6 Fitur", f"{k}_sel") for k in kernels] \
  + [("KNN Semua Fitur", "knn_all"),
     ("Naive Bayes Semua Fitur", "nb_all"),
     ("Decision Tree Semua Fitur", "dt_all"),
     ("KNN 6 Fitur", "knn_sel"),
     ("Naive Bayes 6 Fitur", "nb_sel"),
     ("Decision Tree 6 Fitur", "dt_sel")]

iter_values = [10, 50, 100, 1000, 3000, 5000]
results_16models = []

def train_eval(pipeline, Xtr, Xte, ytr, yte, name, max_iter):
    # untuk model yang tidak punya max_iter (KNN, NB, DT)
    if "classifier__max_iter" in pipeline.get_params():
        pipeline.set_params(classifier__max_iter=max_iter)
    pipeline.fit(Xtr, ytr)
    y_pred = pipeline.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    results_16models.append({
        "model": name,
        "max_iter": max_iter,
        "accuracy": acc
    })

# === Loop model ===
for name, code in model_configs:
    for m_iter in iter_values:
        if code == "baseline_all":
            pipe = Pipeline([
                ("classifier", SVC(kernel="linear", probability=True, random_state=42))
            ])
            train_eval(pipe, X_train_enc, X_test_enc, y_train_enc, y_test_enc, name, m_iter)


        elif code == "baseline_sel":
            pipe = Pipeline([
                ("classifier", SVC(kernel="linear", probability=True, random_state=42))
            ])
            train_eval(pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test, name, m_iter)

        elif code.endswith("_all") and any(k in code for k in kernels):
            k = code.replace("_all", "")
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", SVC(kernel=k, probability=True, random_state=42))
            ])
            train_eval(pipe, X_train, X_test, y_train, y_test, name, m_iter)

        elif code.endswith("_sel") and any(k in code for k in kernels):
            k = code.replace("_sel", "")
            pipe = ImbPipeline([
                ("scaler", scaler),
                ("smote", SMOTE_CFG),
                ("classifier", SVC(kernel=k, probability=True, random_state=42))
            ])
            train_eval(pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test, name, m_iter)

        elif code == "knn_all":
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", KNeighborsClassifier(n_neighbors=5))
            ])
            train_eval(pipe, X_train, X_test, y_train, y_test, name, m_iter)

        elif code == "nb_all":
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", GaussianNB())
            ])
            train_eval(pipe, X_train, X_test, y_train, y_test, name, m_iter)

        elif code == "dt_all":
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", DecisionTreeClassifier(random_state=42))
            ])
            train_eval(pipe, X_train, X_test, y_train, y_test, name, m_iter)

        elif code == "knn_sel":
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", KNeighborsClassifier(n_neighbors=5))
            ])
            train_eval(pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test, name, m_iter)

        elif code == "nb_sel":
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", GaussianNB())
            ])
            train_eval(pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test, name, m_iter)

        elif code == "dt_sel":
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", DecisionTreeClassifier(random_state=42))
            ])
            train_eval(pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test, name, m_iter)

# === Plot hasil ===
df_16models = pd.DataFrame(results_16models)
print("\n=== Ringkasan Akurasi 16 Model pada Tiap maksimum iterasi ===")
print(df_16models.pivot(index="max_iter", columns="model", values="accuracy"))

plt.figure(figsize=(14,8))
print("\n=== Nilai Akurasi per Model & Iterasi ===")
for m in df_16models["model"].unique():
    subset = df_16models[df_16models["model"] == m]
    plt.plot(subset["max_iter"], subset["accuracy"], marker="o", linewidth=2, label=m)
    # Cetak ke terminal
    for _, row in subset.iterrows():
        print(f"{m} | iter={row['max_iter']:<5} | acc={row['accuracy']:.4f}")

plt.xscale("log")
plt.xticks(iter_values, iter_values)
plt.xlabel("Max Iterations (log scale)")
plt.ylabel("Accuracy")
plt.title("Perbandingan Akurasi vs Max Iter\n(16 Model: SVM, KNN, NB, DT)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(True, which="both", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("perbandingan_akurasi_16model_vs_iter.png", dpi=300)
plt.show()

# === Simpan hasil ke file Excel ===
output_excel = "hasil_uji_variabel_max_iter_16model.xlsx"
with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
    # Simpan tabel mentah
    df_16models.to_excel(writer, index=False, sheet_name="Raw Data")

    # Simpan tabel pivot (akurat untuk perbandingan antar model)
    pivot_df = df_16models.pivot(index="max_iter", columns="model", values="accuracy")
    pivot_df.to_excel(writer, sheet_name="Pivot Accuracy")

print(f"\n Hasil uji variasi max_iter (16 model total) berhasil disimpan ke file Excel: {output_excel}")

# ========================================================
# Pastikan variabel subset 6 fitur ini ADA dan konsisten
# ========================================================
X_selected = X_baseline[selected_features].copy()

# =====================================================================
# 13. TAMBAHAN: KNN, Naive Bayes, Decision Tree (+ grafik metrik lengkap)
# =====================================================================
print("\n" + "="*25 + " TAMBAHAN: KNN, NAIVE BAYES, DECISION TREE " + "="*25)

# Model dengan SEMUA FITUR (pakai preprocessor)
extra_models_all = {
    "KNN (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "NaiveBayes (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", GaussianNB())
    ]),
    "DecisionTree (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
}

# Model dengan 6 FITUR TERPILIH (sudah numerik dari LabelEncoder)
extra_models_sel = {
    "KNN (6feat)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),      # KNN diuntungkan dengan scaling
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "NaiveBayes (6feat)": ImbPipeline([
        ("smote", SMOTE_CFG),
        ("classifier", GaussianNB())
    ]),
    "DecisionTree (6feat)": ImbPipeline([
        ("smote", SMOTE_CFG),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
}

# Latih & evaluasi
metrics_rows = []

def eval_and_collect(name, pipe, Xtr, Xte, ytr, yte):
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, y_pred, average="macro", zero_division=0)
    print(f"{name} -> Acc: {acc:.4f} | Prec: {pr:.4f} | Rec: {rc:.4f} | F1: {f1:.4f}")
    metrics_rows.append({"model": name, "accuracy": acc, "precision": pr, "recall": rc, "f1": f1})
    return pipe

trained_extra_models = {}

# Semua fitur
for name, pipe in extra_models_all.items():
    trained = eval_and_collect(name, pipe, X_train, X_test, y_train, y_test)
    trained_extra_models[name] = trained

# 6 fitur terpilih
for name, pipe in extra_models_sel.items():
    trained = eval_and_collect(name, pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test)
    trained_extra_models[name] = trained

# Tambahkan juga SVM terbaik (untuk ikut di grafik gabungan)
svm_best_pipe = ImbPipeline([
    ("scaler", StandardScaler()),
    ("smote", SMOTE_CFG),
    ("classifier", SVC(kernel=best_kernel, probability=True, random_state=42))
])
svm_best_pipe.fit(X_train, y_train)
y_pred_best = svm_best_pipe.predict(X_test)
acc_best = accuracy_score(y_test, y_pred_best)
pr_best, rc_best, f1_best, _ = precision_recall_fscore_support(y_test, y_pred_best, average="macro", zero_division=0)
metrics_rows.append({"model": f"SVM-{best_kernel} (all)", "accuracy": acc_best, "precision": pr_best, "recall": rc_best, "f1": f1_best})
trained_extra_models[f"SVM-{best_kernel} (all)"] = svm_best_pipe

# Buat DataFrame metrik semua model tambahan + SVM best
df_extra_metrics = pd.DataFrame(metrics_rows).set_index("model").sort_values("accuracy", ascending=False)
print("\n=== Ringkasan Metrik: KNN / NB / DT (+ SVM terbaik) ===")
print(df_extra_metrics)

# Gambar grafik komparasi dengan angka di atas batang untuk tiap metrik
def plot_metric_bars(df_metric: pd.DataFrame, metric_name: str, title: str):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_metric.index, df_metric[metric_name])
    plt.title(title)
    plt.ylabel(metric_name.capitalize())
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20, ha="right")
    # angka di atas batang
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 0.01, f"{h*100:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    plt.show()

for metric in ["accuracy", "precision", "recall", "f1"]:
    plot_metric_bars(df_extra_metrics, metric, f"Perbandingan {metric.capitalize()} - KNN / NB / DT (+ SVM terbaik)")

# ========================================================
# 14. SIMPAN SEMUA MODEL KE .pkl
#     - Versi SEMUA FITUR: KNN/NB/DT/SVM kernels
#     - Versi 3 FITUR: KNN/NB/DT/SVM kernels + baseline
# ========================================================
print("\n" + "="*25 + " SIMPAN MODEL (.pkl) " + "="*25)
saved_files = []

# Pastikan X_selected ada
X_selected = X_baseline[selected_features].copy()
# SVM: semua kernel (all features)
for k in kernels:
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel=k, probability=True, random_state=42))
    ])
    pipe.fit(X, y)  # train ke seluruh data
    fname = f"svm_{k}_all.pkl"
    joblib.dump(pipe, fname)
    saved_files.append(fname)

# SVM: 6 fitur terpilih
for k in kernels:
    pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel=k, probability=True, random_state=42))
    ])
    pipe.fit(X_baseline[selected_features], y)
    fname = f"svm_{k}_6fitur.pkl"
    joblib.dump(pipe, fname)
    saved_files.append(fname)

# Baseline (all features & 6fitur)
# Baseline (all features) = SVM murni di X_baseline
base_all = Pipeline([
    ("classifier", SVC(kernel="linear", probability=True, random_state=42))
])
base_all.fit(X_baseline, y)
fname_all = "baseline_svm_all.pkl"
joblib.dump(base_all, fname_all)
saved_files.append(fname_all)


base_sel = Pipeline([
    ("classifier", SVC(kernel="linear", probability=True, random_state=42))
])
base_sel.fit(X_selected, y)     # FIX: pakai X_selected yang valid
fname_sel = "baseline_6fitur.pkl"
joblib.dump(base_sel, fname_sel)
saved_files.append(fname_sel)

# KNN/NB/DT: 6 fitur
for name, pipe in extra_models_sel.items():
    pipe.fit(X_baseline[selected_features], y)
    short = name.split()[0].lower()
    fname = f"{short}_6fitur.pkl"
    joblib.dump(pipe, fname); saved_files.append(fname)

# SVM terbaik (redundan tapi biar eksplisit)
svm_best_all_name = f"svm_{best_kernel}_best_all.pkl"
joblib.dump(svm_best_pipe, svm_best_all_name); saved_files.append(svm_best_all_name)

print("\n=== File model yang disimpan (.pkl) ===")
for f in saved_files:
    print("-", f)

print("\nSelesai. Semua model tambahan (KNN, Naive Bayes, Decision Tree) sudah dievaluasi & disimpan.")


# ========================================================
# 15. PERBANDINGAN SEMUA MODEL (Baseline + Semua Kernel + KNN + NB + DT)
# ========================================================

print("\n" + "="*25 + " PERBANDINGAN SEMUA MODEL (AKURASI, PRESISI, RECALL, F1) " + "="*25)

from sklearn.metrics import precision_recall_fscore_support

all_model_entries = []

def eval_model(name, pipe, Xtr, Xte, ytr, yte, suffix=""):
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, y_pred, average="macro", zero_division=0)
    entry = {"model": f"{name} {suffix}".strip(), "accuracy": acc, "precision": pr, "recall": rc, "f1": f1}
    all_model_entries.append(entry)

# ==== BASELINE ====
eval_model("Baseline", baseline_svm, X_train_enc, X_test_enc, y_train_enc, y_test_enc, "(all)")
eval_model("Baseline", baseline_svm, X_sel_train, X_sel_test, y_sel_train, y_sel_test, "(6fitur)")

# ==== SVM ====
for k in kernels:
    pipe_all = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel=k, probability=True, random_state=42))
    ])
    eval_model(f"SVM-{k}", pipe_all, X_train, X_test, y_train, y_test, "(all)")

    pipe_sel = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel=k, probability=True, random_state=42))
    ])
    eval_model(f"SVM-{k}", pipe_sel, X_sel_train, X_sel_test, y_sel_train, y_sel_test, "(6fitur)")

# ==== KNN / NB / DT ====
for name, pipe in extra_models_all.items():
    eval_model(name.split()[0], pipe, X_train, X_test, y_train, y_test, "(all)")
for name, pipe in extra_models_sel.items():
    eval_model(name.split()[0], pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test, "(6fitur)")

# ==== DataFrame Semua Model ====
df_all = pd.DataFrame(all_model_entries)
df_all["mean_score"] = df_all[["accuracy","precision","recall","f1"]].mean(axis=1)
df_all_sorted = df_all.sort_values("mean_score", ascending=False).reset_index(drop=True)
df_all_sorted.index = df_all_sorted.index + 1  # ranking mulai dari 1

print("\n=== RANKING SEMUA MODEL (TERURUT) ===")
print(df_all_sorted[["accuracy","precision","recall","f1","mean_score"]])

# Ambil model terbaik absolut berdasarkan mean_score
best_row = df_all_sorted.iloc[0]
best_model_name = best_row["model"]

print("Model terbaik absolut:", best_model_name)
print(best_row)

# ========================================================
# Grafik Semua Metrik
# ========================================================
metrics = ["accuracy","precision","recall","f1"]

for metric in metrics:
    plt.figure(figsize=(12,7))
    bars = plt.bar(df_all_sorted["model"], df_all_sorted[metric], color="teal")
    plt.title(f"Perbandingan {metric.capitalize()} Semua Model")
    plt.ylabel(metric.capitalize())
    plt.xticks(rotation=75, ha="right")
    plt.ylim(0, 1.05)
    # Tampilkan nilai vertikal
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, h + 0.005, f"{h:.3f}",
                 ha='center', va='bottom', fontsize=8, rotation=90, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ========================================================
# 17. EVALUASI K-FOLD CROSS VALIDATION (SEMUA FITUR)
#     UNTUK 3 MODEL TERBAIK BERDASARKAN RANKING
# ========================================================
from sklearn.model_selection import KFold, cross_val_score

print("\n" + "="*25 + " K-FOLD CROSS VALIDATION (3 MODEL TERBAIK, SEMUA FITUR) " + "="*25)

# Ambil 3 model teratas dari ranking
top3 = df_all_sorted.head(3)
print("\n=== 3 Model Terbaik Berdasarkan Mean Score ===")
print(top3[["model", "mean_score"]])

# Definisikan K-Fold (misal 5-fold)
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

kfold_results = []

for idx, row in top3.iterrows():
    model_name = row["model"]

    # default: pakai X mentah + preprocessor
    X_used = X
    y_used = y

    # buat instance pipeline berdasarkan nama model
    if "svm" in model_name.lower():
        kernel = "linear"
        if "rbf" in model_name.lower():
            kernel = "rbf"
        elif "poly" in model_name.lower():
            kernel = "poly"
        elif "sigmoid" in model_name.lower():
            kernel = "sigmoid"

        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
        ])

    elif "knn" in model_name.lower():
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ])

    elif "naive" in model_name.lower():
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", GaussianNB())
        ])

    elif "decision" in model_name.lower():
        pipe = ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", DecisionTreeClassifier(random_state=42))
        ])

    elif "baseline" in model_name.lower():
        # ❗ Baseline pakai data yang sudah di-LabelEncoder (X_encoded)
        pipe = Pipeline([
            ("classifier", SVC(kernel="linear", probability=True, random_state=42))
        ])
        X_used = X_baseline   # <--- ini kuncinya
        y_used = y           # label sudah di-encode sebelumnya

    else:
        continue  # skip model tidak dikenali

    # Evaluasi k-fold cross validation (akurasi)
    scores = cross_val_score(pipe, X_used, y_used, cv=kf, scoring='accuracy')
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    kfold_results.append({
        "model": model_name,
        "k": k,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc
    })
    print(f"\nModel: {model_name}")
    print(f"{k}-Fold Accuracy: {scores}")
    print(f"Rata-rata: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Rata-rata: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")


# Buat DataFrame hasil K-Fold
df_kfold = pd.DataFrame(kfold_results)
print("\n=== Ringkasan K-Fold Cross Validation (3 Model Terbaik) ===")
print(df_kfold)

svm_poly = np.array([0.88461538, 0.92307692, 0.96, 0.88, 0.92])
svm_rbf  = np.array([0.88461538, 0.88461538, 0.96, 0.92, 0.72])
knn      = np.array([0.84615385, 0.80769231, 0.88, 0.92, 0.72])

stack = np.vstack([svm_poly, svm_rbf, knn])  # shape (3,5)
mean_fold = stack.mean(axis=0)
std_fold  = stack.std(axis=0)

for i in range(5):
    print(f"Fold {i+1}: mean={mean_fold[i]:.4f}, std={std_fold[i]:.4f}")


# Simpan ke CSV untuk tabel di laporan
df_kfold.to_csv("kfold_3model_terbaik.csv", index=False)

# Visualisasi
plt.figure(figsize=(7,5))
plt.bar(df_kfold["model"], df_kfold["mean_accuracy"], yerr=df_kfold["std_accuracy"],
        capsize=5, color="skyblue", edgecolor="black")
plt.ylabel("Mean Accuracy")
plt.title(f"Perbandingan {k}-Fold Cross Validation (3 Model Terbaik, Semua Fitur)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

from sklearn.model_selection import KFold

def build_pipeline_from_model_name(name):
    lname = name.lower()

    if "svm-poly" in lname:
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(kernel="poly", probability=True, random_state=42))
        ])

    if "svm-rbf" in lname:
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
        ])

    if "svm-linear" in lname:
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(kernel="linear", probability=True, random_state=42))
        ])

    if "knn" in lname:
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ])

    if "decision" in lname:
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42))
        ])

    raise ValueError(f"Model tidak dikenali: {name}")


print("\n" + "="*25 + f" DATA YANG SALAH PREDIKSI - {best_model_name} (5-FOLD) " + "="*25)

best_model_name = top3_models[0]
print(f"\nAnalisis data salah prediksi untuk model terbaik: {best_model_name}")

pipe_best = build_pipeline_from_model_name(best_model_name)

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows_salah = []
fold_no = 1
for train_idx, test_idx in kf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    pipe_best.fit(X_tr, y_tr)
    y_pred = pipe_best.predict(X_te)

    salah_mask = (y_pred != y_te)
    idx_salah = test_idx[salah_mask]

    print(f"Fold {fold_no}: jumlah salah prediksi = {len(idx_salah)}")

    for pos, y_t, y_p in zip(idx_salah, y_te[salah_mask], y_pred[salah_mask]):
        rows_salah.append({
            "fold": fold_no,
            "index_df_clean": int(pos),
            "kelas_asli": str(y_t),
            "kelas_pred": str(y_p)
        })

    fold_no += 1

df_salah = pd.DataFrame(rows_salah)
print("\nRingkasan data salah prediksi:")
print(df_salah)


# kalau mau lihat fitur lengkap dari beberapa baris salah:
print("\n=== CONTOH 10 BARIS DATA ASLI YANG SALAH PREDIKSI ===")
print(df_clean.loc[df_salah["index_df_clean"].unique()[:10]])

# kalau mau simpan ke Excel:
df_salah.to_excel("data_salah_prediksi_modelterbaik_5fold.xlsx", index=False)

# ------------------------------------------------------------
# GABUNGKAN DENGAN df_clean AGAR SEMUA FITUR IKUT TEREKAM
# ------------------------------------------------------------
# reset index df_clean supaya index asli jadi kolom 'index_df_clean'
df_clean_reset = df_clean.reset_index().rename(columns={"index": "index_df_clean"})

# merge meta-info (fold, kelas_asli/pred, dll) dengan seluruh kolom fitur asli
df_salah_full = df_salah.merge(
    df_clean_reset,
    on="index_df_clean",
    how="left"
)

print("\n=== SEMUA DATA YANG SALAH PREDIKSI (Model Terbaik, 5-FOLD, LENGKAP DENGAN FITUR) ===")
print(df_salah_full)

# ------------------------------------------------------------
# TAMPILKAN OUTPUT PER FOLD DI TERMINAL
# ------------------------------------------------------------
for f in sorted(df_salah_full["fold"].unique()):
    subset = df_salah_full[df_salah_full["fold"] == f]
    print(f"\n=========== FOLD {f} - {len(subset)} DATA SALAH PREDIKSI ===========")
    # kalau mau semua baris:
    print(subset)
    # kalau dirasa terlalu panjang di console, bisa pakai:
    # print(subset.head(20))

# ------------------------------------------------------------
# SIMPAN KE CSV & EXCEL
# ------------------------------------------------------------
df_salah_full.to_csv(
    "semua_data_salah_prediksi_svm_poly_5fold.csv",
    index=False
)
df_salah_full.to_excel(
    "semua_data_salah_prediksi_svm_poly_5fold.xlsx",
    index=False
)

print("\n File detail data salah prediksi tersimpan sebagai:")
print("- semua_data_salah_prediksi_svm_poly_5fold.csv")
print("- semua_data_salah_prediksi_svm_poly_5fold.xlsx")


# ------------------------------------------------------------
# 19b. K-FOLD CROSS VALIDATION (6 MODEL TERBAIK - SELECTED FEATURES)
# ------------------------------------------------------------

print("\n" + "="*25 + " K-FOLD CROSS VALIDATION (6 MODEL TERBAIK - FITUR TERPILIH) " + "="*25)

# 1) Ambil hanya model yang kategori 6 fitur dari ranking
mask_sel = df_all_sorted["model"].str.contains("6fitur", case=False)
df_6fitur = df_all_sorted[mask_sel].copy()

print("\n=== SEMUA MODEL KATEGORI 6 FITUR (URUT DARI TERBAIK) ===")
print(df_6fitur[["model", "mean_score"]])

top_n = min(3, len(df_6fitur))  # kalau misalnya baru ada 2 model, ambil 2 saja
top3_sel = df_6fitur.head(top_n).reset_index(drop=True)

print(f"\n=== {top_n} Model Terbaik (Kategori 6 Fitur Terpilih) ===")
print(top3_sel[["model", "mean_score"]])

# 2) Data khusus 6 fitur terpilih
X_sel_all = X_baseline[selected_features]
y_all = y

# 3) Siapkan KFold
k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


kfold_results_sel = []
model_scores_sel = []   # untuk menyimpan skor per-fold setiap model

# 4) Loop setiap model dalam top3_sel
for _, row in top3_sel.iterrows():
    model_name_full = row["model"]      # contoh: "SVM-rbf (6fitur)"
    model_name = model_name_full.lower()

    # Identifikasi model dan buat pipeline yg sesuai
    if "svm" in model_name:
        if "rbf" in model_name:
            kernel = "rbf"
        elif "poly" in model_name:
            kernel = "poly"
        elif "sigmoid" in model_name:
            kernel = "sigmoid"
        else:
            kernel = "linear"

        pipe = ImbPipeline([
            ("smote", SMOTE_CFG),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
        ])

    elif "knn" in model_name:
        pipe = ImbPipeline([
            ("smote", SMOTE_CFG),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ])

    elif "naive" in model_name:  # NaiveBayes
        pipe = ImbPipeline([
            ("smote", SMOTE_CFG),
            ("classifier", GaussianNB())
        ])

    elif "decision" in model_name:
        pipe = ImbPipeline([
            ("smote", SMOTE_CFG),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ])

    elif "baseline" in model_name:
        pipe = Pipeline([
            ("classifier", SVC(kernel="linear", probability=True, random_state=42))
        ])

    else:
        print(f"Model '{model_name_full}' tidak dikenali → dilewati.")
        continue

    pipe = ImbPipeline([
        ("smote", SMOTE_CFG),
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
    ])
    # 5) Evaluasi K-Fold
    scores = cross_val_score(pipe, X_sel_all, y_all, cv=kf, scoring='accuracy')
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)

    kfold_results_sel.append({
        "model": model_name_full,
        "k": k,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc
    })

    model_scores_sel.append({
        "model": model_name_full,
        "scores": scores
    })

    print(f"\nModel: {model_name_full}")
    print(f"{k}-Fold Accuracy: {scores}")
    print(f"Rata-rata: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Rata-rata: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# 6) Ringkasan per-model
df_kfold_sel = pd.DataFrame(kfold_results_sel)
print("\n=== Ringkasan K-Fold (3 Model Terbaik - Fitur Terpilih) ===")
print(df_kfold_sel)

# 7) Tabel setiap lipatan + mean & std antar model (per fold)
if model_scores_sel:
    stack = np.vstack([m["scores"] for m in model_scores_sel])
    fold_idx = np.arange(1, k + 1)

    mean_fold_sel = stack.mean(axis=0)
    std_fold_sel  = stack.std(axis=0)

    data_fold = {"Fold": fold_idx}
    for m in model_scores_sel:
        data_fold[m["model"]] = m["scores"]
    data_fold["Akurasi Rata-Rata"] = mean_fold_sel
    data_fold["Nilai Standar Deviasi Antar Model"] = std_fold_sel

    df_kfold_sel_fold = pd.DataFrame(data_fold)

    print("\n=== Tabel Setiap Lipatan K-Fold (3 Model Terbaik - 3 Fitur Terpilih) ===")
    print(df_kfold_sel_fold)

    for i in range(k):
        print(f"Fold {i+1}: mean={mean_fold_sel[i]:.4f}, std={std_fold_sel[i]:.4f}")

    # Simpan tabel per-fold
    df_kfold_sel_fold.to_csv(
        "kfold_3model_terbaik_selected_features_perfold.csv",
        index=False
    )

# 8) Simpan ringkasan per-model ke CSV
df_kfold_sel.to_csv("kfold_3model_terbaik_selected_features.csv", index=False)

# 9) Plot visualisasi
plt.figure(figsize=(7,5))
plt.bar(df_kfold_sel["model"], df_kfold_sel["mean_accuracy"],
        yerr=df_kfold_sel["std_accuracy"], capsize=5,
        color="salmon", edgecolor="black")
plt.ylabel("Mean Accuracy (3 Fitur)")
plt.title(f"{k}-Fold Cross Validation (Top-{top_n} Model - Selected Features)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# ========================================================
# 18. Grafik Gabungan seperti Fig. 2 (Final Compact & Low Height Version)
# ========================================================
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

print("\n" + "="*25 + " GRAFIK GABUNGAN (ALL vs 3 FITUR) DENGAN TRAINING TIME " + "="*25)

# ------------------------------------------------------------
# Fungsi bantu: latih model + ukur waktu + hitung metrik
# ------------------------------------------------------------
def get_metrics_with_time(model_name, pipeline, Xtr, Xte, ytr, yte):
    start = time.time()
    pipeline.fit(Xtr, ytr)
    train_time = time.time() - start

    y_pred = pipeline.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        yte, y_pred, average="macro", zero_division=0
    )

    return {
        "model": model_name,
        "accuracy": acc,
        "precision": pr,
        "recall": rc,
        "f1": f1,
        "training_time": train_time
    }


# ------------------------------------------------------------
# Daftar model (All Features)
# ------------------------------------------------------------
model_dict_all = {
    "Baseline (all)": Pipeline([
        ("classifier", SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "SVM-linear (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "SVM-rbf (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "SVM-poly (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="poly", probability=True, random_state=42))
    ]),
    "SVM-sigmoid (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="sigmoid", probability=True, random_state=42))
    ]),
    "KNN (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "NaiveBayes (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", GaussianNB())
    ]),
    "DecisionTree (all)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", DecisionTreeClassifier(random_state=42))
    ])
}

# ------------------------------------------------------------
# Daftar model (6 Fitur Terpilih)
# ------------------------------------------------------------
model_dict_sel = {
    "Baseline (6fitur)": Pipeline([
        ("classifier", SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "SVM-linear (6fitur)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="linear", probability=True, random_state=42))
    ]),
    "SVM-rbf (6fitur)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "SVM-poly (6fitur)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="poly", probability=True, random_state=42))
    ]),
    "SVM-sigmoid (6fitur)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="sigmoid", probability=True, random_state=42))
    ]),
    "KNN (6fitur)": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "NaiveBayes (6fitur)": ImbPipeline([
        ("smote", SMOTE_CFG),
        ("classifier", GaussianNB()),
    ]),
    "DecisionTree (6fitur)": ImbPipeline([
        ("smote", SMOTE_CFG),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ])
}

# ------------------------------------------------------------
# Evaluasi semua model
# ------------------------------------------------------------
print("\nMenghitung metrik dan training time (All Features)...")
metrics_all = []
for name, pipe in model_dict_all.items():
    if "Baseline (all)" in name:
        # baseline pakai data yang sudah di-LabelEncoder (semua numerik)
        m = get_metrics_with_time(name, pipe,
                                  X_train_enc, X_test_enc,
                                  y_train_enc, y_test_enc)
    else:
        # model lain sudah punya preprocessor sendiri
        m = get_metrics_with_time(name, pipe,
                                  X_train, X_test,
                                  y_train, y_test)
    metrics_all.append(m)


print("\nMenghitung metrik dan training time (Selected Features)...")
metrics_sel = [get_metrics_with_time(name, pipe, X_sel_train, X_sel_test, y_sel_train, y_sel_test)
               for name, pipe in model_dict_sel.items()]

df_allfeat = pd.DataFrame(metrics_all)
df_selfeat = pd.DataFrame(metrics_sel)

# ------------------------------------------------------------
# Fungsi plot (grafik rendah + tabel muat penuh)
# ------------------------------------------------------------
def plot_overall_performance(df, title, filename):
    metrics = ["accuracy", "precision", "recall", "f1", "training_time"]
    models = df["model"].tolist()

    # Grafik lebih rendah agar tabel muat
    fig, ax = plt.subplots(figsize=(9.5, 4.5))

    x = np.arange(len(metrics))
    width = 0.08
    colors = plt.cm.tab10.colors

    for i, model in enumerate(models):
        ax.bar(x + (i - len(models)/2)*width, df.loc[i, metrics],
               width, label=model, color=colors[i % len(colors)])

    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 5)
    ax.set_xticks(x)
    ax.set_xticklabels([""] * len(metrics))
    ax.set_title(title, pad=12, fontsize=10, weight="bold")
    ax.legend(loc="upper right", fontsize=6.5, ncol=2, frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Tabel proporsional
    table_vals = np.round(df[metrics].values, 3)
    row_labels = [m.replace("(all)", "").replace("(6fitur)", "").strip() for m in df["model"]]
    col_labels = [m.upper().replace("_", " ") for m in metrics]

    table = plt.table(cellText=table_vals,
                      rowLabels=row_labels,
                      colLabels=col_labels,
                      loc="bottom",
                      cellLoc="center")

    table.auto_set_font_size(False)
    table.set_fontsize(5.5)
    table.scale(0.65, 0.75)

    # Ruang bawah diperbesar agar tabel muat
    plt.subplots_adjust(left=0.1, right=0.95, top=0.8, bottom=0.55)
    plt.tight_layout(pad=1.2)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\n=== {title} ===")
    print(df[["model", "accuracy", "precision", "recall", "f1", "training_time"]]
          .to_string(index=False, float_format="%.6f"))

# ------------------------------------------------------------
# Jalankan plotting
# ------------------------------------------------------------
plot_overall_performance(
    df_allfeat,
    "Overall Feature Classification Performance Results (All Features)",
    "overall_performance_allfeat_fig2_final_low.png"
)

plot_overall_performance(
    df_selfeat,
    "Overall Feature Classification Performance Results (Selected Features)",
    "overall_performance_selfeat_fig2_final_low.png"
)

print("\nFile grafik tersimpan sebagai:")
print("- overall_performance_allfeat_fig2_final_low.png")
print("- overall_performance_selfeat_fig2_final_low.png")

# ========================================================
# 17b. K-FOLD CROSS VALIDATION (SEMUA MODEL - SEMUA FITUR)
# ========================================================
from sklearn.model_selection import KFold, cross_val_score

print("\n" + "="*25 + " K-FOLD CROSS VALIDATION (SEMUA MODEL - SEMUA FITUR) " + "="*25)

k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


kfold_all_models = []

for name, pipe in model_dict_all.items():
    # Baseline (all) pakai X_baseline (sudah di-LabelEncoder, semua numerik)
    if "Baseline (all)" in name:
        X_used = X_baseline
        y_used = y
    else:
        # Model lain sudah di-wrap dengan preprocessor, pakai X mentah
        X_used = X
        y_used = y

    scores = cross_val_score(pipe, X_used, y_used, cv=kf, scoring="accuracy")
    mean_acc = scores.mean()
    std_acc = scores.std()

    kfold_all_models.append({
        "model": name,
        "k": k,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc
    })

    print(f"\nModel: {name}")
    print(f"{k}-Fold Accuracy: {scores}")
    print(f"Rata-rata: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Rata-rata: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

# Jadikan DataFrame untuk ringkasan
df_kfold_all_models = pd.DataFrame(kfold_all_models)
print("\n=== Ringkasan K-Fold Cross Validation (Semua Model - Semua Fitur) ===")
print(df_kfold_all_models)

# --------------------------------------------------------
# Grafik perbandingan (mirip gaya gambar, berbasis mean accuracy)
# --------------------------------------------------------
plt.figure(figsize=(9, 5))
bars = plt.bar(df_kfold_all_models["model"],
               df_kfold_all_models["mean_accuracy"],
               yerr=df_kfold_all_models["std_accuracy"],
               capsize=5)

plt.ylabel("Mean Accuracy (5-Fold)")
plt.ylim(0, 1.05)
plt.title("Perbandingan K-Fold Cross Validation\nSemua Model (All Features)")
plt.xticks(rotation=25, ha="right")

# Tampilkan nilai akurasi di atas batang
for bar, acc in zip(bars, df_kfold_all_models["mean_accuracy"]):
    h = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0,
             h + 0.01,
             f"{acc*100:.1f}%",
             ha="center", va="bottom", fontsize=8, fontweight="bold")

plt.tight_layout()
plt.savefig("kfold_all_models_all_features.png", dpi=300)
plt.show()

print("\nFile grafik K-Fold semua model tersimpan sebagai: kfold_all_models_all_features.png")


# ========================================================
# 20. GRAFIK FITUR TERPILIH 3 MODEL TERBAIK (POLY, RBF, KNN)
#     - Menggunakan permutation importance pada X_encoded
#     - 3 model terbaik: SVM-poly, SVM-rbf, KNN (tanpa seleksi fitur di awal)
# ========================================================

print("\n" + "="*25 + " FEATURE IMPORTANCE 3 MODEL TERBAIK (POLY, RBF, KNN) " + "="*25)

feature_names = X.columns
n_features_total = len(feature_names)

# Pakai split yang sudah ada (X_train_enc, X_test_enc, y_train_enc, y_test_enc)
best_models_feat = {
    "SVM-poly": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="poly", probability=True, random_state=42))
    ]),
    "SVM-rbf": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
    "SVM-linear": ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE_CFG),
        ("classifier", SVC(kernel="linear", probability=True, random_state=42))
    ])
}


importance_results = {}
feature_ranking = {}

print("\n" + "="*25 + " FEATURE IMPORTANCE 3 MODEL TERBAIK (HOLDOUT) " + "="*25)

for model_name in top3_models_holdout:
    print(f"\nMenghitung permutation importance untuk {model_name} ...")

    pipe = build_pipeline_from_name(model_name)
    pipe.fit(X_train, y_train)

    r = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring="accuracy"
    )

    imp_mean = r.importances_mean
    idx_sorted = np.argsort(np.abs(imp_mean))[::-1]

    print(f"Top 7 fitur untuk {model_name}:")
    for i in idx_sorted[:7]:
        print(f" {X.columns[i]} -> importance_mean={imp_mean[i]:.5f}")
    name = model_name.lower().replace(" ", "-")
    importance_results[name] = imp_mean
    # ranking pakai nilai absolut (fitur dengan pengaruh terbesar)
    feature_ranking[name] = np.argsort(np.abs(imp_mean))[::-1]

    # Plot top-N fitur (misal 15 fitur teratas atau semua kalau <15)
    top_n = min(15, n_features_total)
    top_idx = feature_ranking[name][:top_n]

    plt.figure(figsize=(10, 4))
    plt.bar(range(top_n), imp_mean[top_idx], align="center")
    plt.xticks(range(top_n), feature_names[top_idx], rotation=75, ha="right")
    plt.ylabel("Permutation Importance (mean)")
    plt.title(f"Grafik Fitur Terpilih ({name})\n(Top {top_n} fitur berdasarkan bobot importance)")
    plt.tight_layout()
    fname_fig = f"feature_importance_{name.replace('-', '').replace(' ', '_').lower()}.png"
    plt.savefig(fname_fig, dpi=300)
    plt.show()

    print(f"Top {top_n} fitur untuk {name}:")
    for rank, idx_f in enumerate(top_idx, start=1):
        print(f"{rank:2d}. {feature_names[idx_f]} -> importance_mean={imp_mean[idx_f]:.5f}")


# ========================================================
# SHAP UNTUK 3 MODEL TERBAIK
# ========================================================
print("\n" + "="*25 + " SHAP UNTUK 3 MODEL TERBAIK (HOLDOUT) " + "="*25)

for model_name in top3_models_holdout:
    print(f"\n>>> Menghitung SHAP untuk {model_name} ...")

    pipe = build_pipeline_from_name(model_name)
    pipe.fit(X_train, y_train)

    if "decision" in model_name.lower():
        explainer = shap.TreeExplainer(pipe.named_steps["classifier"])
        shap_values = explainer.shap_values(X_exp)
    else:
        def predict_fn(x):
            return pipe.predict_proba(pd.DataFrame(x, columns=X.columns))

        explainer = shap.KernelExplainer(predict_fn, X_bg)
        shap_values = explainer.shap_values(X_exp)

    imp = mean_abs_shap(shap_values, n_features_expected=len(X.columns))

    df_shap = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": imp
    }).sort_values("mean_abs_shap", ascending=False)

    print(df_shap.head(7))

shap_importance_dict = {}

fitur_shap = [
    "MINYAK (mPa.s)",
    "JUMLAH MINYAK (g)",
    "SURFAKTAN (HLB)",
    "JUMLAH SURFAKTAN (g)",
    "FASA AIR (V)",
    "JUMLAH FASA AIR (g)",
    "Ko-Surfaktan (Rasio)"
]
print("\nFitur yang dipakai untuk SHAP:", fitur_shap)

# ===== Preprocessor khusus fitur_shap saja =====
numeric_cols_shap = [c for c in numeric_cols if c in fitur_shap]


num_trans_shap = Pipeline([("scaler", StandardScaler())])

preprocessor_shap = ColumnTransformer(
    transformers=[
        ("num", num_trans_shap, numeric_cols_shap)
    ],
    remainder="drop"
)

# ===== Background & data yang dijelaskan =====
bg_size = min(80, len(X_train))
exp_size = min(40, len(X_test))

X_bg  = X_train[fitur_shap].sample(bg_size, random_state=0)
X_exp = X_test[fitur_shap].sample(exp_size, random_state=0)

def to_df(x, cols):
    if isinstance(x, np.ndarray):
        return pd.DataFrame(x, columns=cols)
    return x


def save_and_plot(name, feature_names, shap_importance):
    # pastikan 1D
    shap_importance = np.array(shap_importance).reshape(-1)

    idx_sorted = np.argsort(shap_importance)[::-1]
    df_shap_all = pd.DataFrame({
        "feature": np.array(feature_names)[idx_sorted],
        "mean_abs_shap": shap_importance[idx_sorted]
    })

    print(f"\n=== Ranking Mean |SHAP| (SEMUA FITUR) untuk {name} ===")
    print(df_shap_all)

    fname_csv = f"shap_importance_{name.replace('-', '').replace(' ', '_').lower()}_allfeatures.csv"
    df_shap_all.to_csv(fname_csv, index=False)
    print(f"File SHAP tersimpan: {fname_csv}")

    n_features = len(feature_names)
    plt.figure(figsize=(max(10, 0.55 * n_features), 4))
    plt.bar(range(n_features), shap_importance[idx_sorted])
    plt.xticks(range(n_features), np.array(feature_names)[idx_sorted], rotation=75, ha="right")
    plt.ylabel("Mean |SHAP|")
    plt.title(f"SHAP Feature Importance - {name}")
    plt.tight_layout()
    fname_fig = f"shap_bar_{name.replace('-', '').replace(' ', '_').lower()}_allfeatures.png"
    plt.savefig(fname_fig, dpi=300)
    plt.show()
    print(f"Grafik SHAP tersimpan: {fname_fig}")

    top_n = min(6, n_features)
    print(f"\nTop {top_n} fitur berdasarkan mean |SHAP| untuk {name}:")
    for i in range(top_n):
        feat = df_shap_all.iloc[i, 0]
        val  = df_shap_all.iloc[i, 1]
        print(f"{i+1:2d}. {feat} -> mean|SHAP|={val:.6f}")

# ========================================================
# (A) SVM-POLY (KernelExplainer) -> fitur = fitur_shap
# ========================================================
print("\n>>> Menghitung Kernel SHAP untuk SVM-linear ...")
pipe_linear = ImbPipeline([
    ("preprocessor", preprocessor_shap),
    ("smote", SMOTE_CFG),
    ("classifier", SVC(kernel="linear", probability=True, random_state=42))
])
pipe_linear.fit(X_train[fitur_shap], y_train)

def predict_poly(x):
    xdf = to_df(x, fitur_shap)
    return pipe_linear.predict_proba(xdf)
expl_linear = shap.KernelExplainer(predict_poly, X_bg)
sv_linear = expl_linear.shap_values(X_exp, nsamples="auto")

imp_linear = mean_abs_shap(sv_linear, n_features_expected=len(fitur_shap))
shap_importance_dict["SVM-linear"] = imp_linear
save_and_plot("SVM-linear", fitur_shap, imp_linear)
# ========================================================
# (B) KNN (KernelExplainer) -> fitur = fitur_shap
# ========================================================
print("\n>>> Menghitung Kernel SHAP untuk SVM-RBF ...")
pipe_rbf = ImbPipeline([
    ("preprocessor", preprocessor_shap),
    ("smote", SMOTE_CFG),
    ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
])
pipe_rbf.fit(X_train[fitur_shap], y_train)

def predict_knn(x):
    xdf = to_df(x, fitur_shap)
    return pipe_rbf.predict_proba(xdf)
expl_rbf = shap.KernelExplainer(predict_knn, X_bg)
sv_rbf = expl_rbf.shap_values(X_exp, nsamples="auto")

imp_rbf = mean_abs_shap(sv_rbf, n_features_expected=len(fitur_shap))
shap_importance_dict["SVM-RBF"] = imp_rbf
save_and_plot("SVM-RBF", fitur_shap, imp_rbf)

# ========================================================
# (C) DECISION TREE (TreeExplainer) -> fitur = fitur_shap (ruang fitur asli)
print("\n>>> Menghitung Tree SHAP untuk DecisionTree (fitur asli) ...")

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train[fitur_shap], y_train)

expl_dt = shap.TreeExplainer(dt)
sv_dt = expl_dt.shap_values(X_exp[fitur_shap])

imp_dt = mean_abs_shap(sv_dt, n_features_expected=len(fitur_shap))
shap_importance_dict["DecisionTree"] = imp_dt
save_and_plot("DecisionTree", fitur_shap, imp_dt)


print("\n SHAP selesai untuk: SVM-linear, SVM-RBF, DecisionTree")


print("\n" + "="*25 + 
      " PERBANDINGAN KINERJA MODEL (ALL vs -10% vs -17% FITUR) " +
      "="*25)

pengurangan_list = [0.10, 0.17]
results_feature_reduction = []

def hitung_metrik_model(pipe, Xtr, Xte, ytr, yte):
    pipe.fit(Xtr, ytr)
    y_pred = pipe.predict(Xte)
    acc = accuracy_score(yte, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        yte, y_pred, average="macro", zero_division=0
    )
    return acc, pr, rc, f1


for name in top3_models_holdout:
    print(f"\n=== Model: {name} ===")

    lname = name.lower()

    if "svm" in lname and "poly" in lname:
        clf = SVC(kernel="poly", probability=True, random_state=42)
    elif "svm" in lname and "rbf" in lname:
        clf = SVC(kernel="rbf", probability=True, random_state=42)
    elif "svm" in lname and "linear" in lname:
        clf = SVC(kernel="linear", probability=True, random_state=42)
    elif "knn" in lname:
        clf = KNeighborsClassifier(n_neighbors=5)
    elif "decision" in lname or "tree" in lname:
        clf = DecisionTreeClassifier(random_state=42)
    else:
        raise ValueError(f"Model tidak dikenali: {name}")


    # -----------------------------
    # 2) ALL FEATURES (preprocessor penuh)
    # -----------------------------
    pipe_all = ImbPipeline(steps=[
        ("scaler", StandardScaler()),   # pakai scaler awal (semua kolom)
        ("smote", SMOTE_CFG),
        ("classifier", clone(clf)),
    ])

    acc_all, pr_all, rc_all, f1_all = hitung_metrik_model(
        pipe_all, X_train, X_test, y_train, y_test
    )

    results_feature_reduction.append({
        "model": name,
        "konfigurasi": "All Fitur",
        "accuracy": acc_all,
        "precision": pr_all,
        "recall": rc_all,
        "f1": f1_all
    })

    print(f"All Fitur -> Acc={acc_all:.4f}, Prec={pr_all:.4f}, Rec={rc_all:.4f}, F1={f1_all:.4f}")

    # -----------------------------
    # 3) PENGURANGAN 10% & 17% FITUR
    #    (preprocessor baru dengan subset kolom)
    # -----------------------------
    # normalisasi nama model (HARUS sama dengan saat feature_ranking dibuat)
    key_name = name.lower().replace(" ", "-")

    if key_name not in feature_ranking:
        raise KeyError(
            f"feature_ranking tidak punya key '{key_name}'. "
            f"Available keys: {list(feature_ranking.keys())}"
        )

    ranking_idx = feature_ranking[key_name]

    for pengurangan in pengurangan_list:
        keep_prop = 1.0 - pengurangan
        keep_count = max(1, int(np.ceil(keep_prop * n_features_total)))
        selected_idx = ranking_idx[:keep_count]
        selected_cols = feature_names[selected_idx]   # nama kolom yang dipertahankan

        # subset data MENTAH (bukan X_encoded) berdasarkan selected_cols
        X_train_red = X_train[selected_cols]
        X_test_red  = X_test[selected_cols]

        # tentukan numeric 
        numeric_cols_red = [c for c in numeric_cols if c in selected_cols]

        num_trans_red = Pipeline([
            ("scaler", StandardScaler())
        ])

        preprocessor_red = ColumnTransformer(
            transformers=[
                ("num", num_trans_red, numeric_cols_red)
            ]
        )

        pipe_red = ImbPipeline(steps=[
            ("preprocessor", preprocessor_red),
            ("classifier", clone(clf))    # clone classifier-nya saja
        ])

        acc_red, pr_red, rc_red, f1_red = hitung_metrik_model(
            pipe_red, X_train_red, X_test_red, y_train, y_test
        )

        label_conf = f"Top {keep_count} Fitur (drop {int(pengurangan*100)}%)"
        results_feature_reduction.append({
            "model": name,
            "konfigurasi": label_conf,
            "accuracy": acc_red,
            "precision": pr_red,
            "recall": rc_red,
            "f1": f1_red
        })

        print(f"{label_conf} -> Acc={acc_red:.4f}, Prec={pr_red:.4f}, Rec={rc_red:.4f}, F1={f1_red:.4f})")


# --------------------------------------------------------
# Buat DataFrame dan simpan ke CSV
# --------------------------------------------------------
df_feat_red = pd.DataFrame(results_feature_reduction)
print("\n=== Ringkasan Kinerja (All vs Pengurangan 10% & 17% Fitur) ===")
print(df_feat_red)

df_feat_red.to_csv("perbandingan_kinerja_3model_pengurangan_fitur.csv", index=False)

# --------------------------------------------------------
# Grafik Perbandingan (per model) – fokus Akurasi
# --------------------------------------------------------
for model_name in df_feat_red["model"].unique():
    subset = df_feat_red[df_feat_red["model"] == model_name]

    plt.figure(figsize=(6,4))
    bars = plt.bar(subset["konfigurasi"], subset["accuracy"])
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(f"Grafik Perbandingan Kinerja Model Fitur Terpilih\n{model_name}")
    plt.xticks(rotation=20, ha="right")

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0,
                 h + 0.01,
                 f"{h*100:.1f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fname_fig = f"perf_reduction_{model_name.replace('-', '').replace(' ', '_').lower()}.png"
    plt.savefig(fname_fig, dpi=300)
    plt.show()

    print(f"Grafik perbandingan kinerja tersimpan sebagai: {fname_fig}")


print("\n" + "="*25 + " K-FOLD DETAIL: SVM LINEAR / RBF / POLY (SEMUA FITUR) " + "="*25)

# 3 model SVM semua fitur yang mau dicek per fold
svm_models_all = {
    "SVM-linear (all)": SVC(kernel="linear", probability=True, random_state=42),
    "SVM-rbf (all)":    SVC(kernel="rbf",    probability=True, random_state=42),
    "SVM-poly (all)":   SVC(kernel="poly",   probability=True, random_state=42),
}

k = 5
kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


rows_detail = []   # untuk DataFrame detail semua fold

for name, clf in svm_models_all.items():
    print(f"\n==== {name} ====")
    fold_acc_list = []

    fold_no = 1
    for train_idx, test_idx in kf.split(X, y):
        # pakai semua fitur mentah + preprocessor, sama seperti model utama
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipe = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("classifier", clone(clf))
        ])

        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0
        )

        fold_acc_list.append(acc)

        rows_detail.append({
            "model": name,
            "fold": fold_no,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        print(f"Fold {fold_no}: "
              f"Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")
        fold_no += 1

    print(f"Rata-rata akurasi {name}: {np.mean(fold_acc_list):.4f}")

# Buat DataFrame semua detail 3 model
df_kfold_svm_all = pd.DataFrame(rows_detail)
print("\n=== DETAIL K-FOLD 3 MODEL SVM (SEMUA FITUR) ===")
print(df_kfold_svm_all)

# Simpan ke Excel supaya mudah isi tabel di laporan
df_kfold_svm_all.to_excel("kfold_detail_svm_3model_allfitur.xlsx", index=False)
print("\n File disimpan: kfold_detail_svm_3model_allfitur.xlsx")

# ========================================================
# 21. ANALISIS 6 FITUR TERPILIH:
#     - Mutual Information
#     - LinearSVC L1
#     - LinearSVC L2
#     - SHAP SVM poly, rbf, linear
# ========================================================

from sklearn.svm import LinearSVC

print("\n" + "="*25 + " ANALISIS BERBASIS 6 FITUR TERPILIH " + "="*25)

# --------------------------------------------------------
# Helper: evaluasi semua model pada subset fitur tertentu
# --------------------------------------------------------
def evaluate_models_on_subset_6feat(selected_features, label_subset):
    """
    selected_features : list nama kolom (harus ada di X_baseline)
    label_subset      : string label untuk judul grafik
    """
    print(f"\n--- Evaluasi subset fitur: {label_subset} ---")
    print("Fitur yang digunakan:", selected_features)

    # Data subset (semua sudah numerik di X_baseline)
    X_sub = X_baseline[selected_features]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_sub, y, test_size=0.2, random_state=42, stratify=y
    )

    models_subset = {
        "Baseline-linear": Pipeline([
            ("classifier", SVC(kernel="linear", probability=True, random_state=42))
        ]),
        "SVM-linear": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel="linear", probability=True, random_state=42))
        ]),
        "SVM-rbf": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
        ]),
        "SVM-poly": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel="poly", probability=True, random_state=42))
        ]),
        "SVM-sigmoid": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", SVC(kernel="sigmoid", probability=True, random_state=42))
        ]),
        "KNN": ImbPipeline([
            ("scaler", StandardScaler()),
            ("smote", SMOTE_CFG),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "NaiveBayes": ImbPipeline([
            ("smote", SMOTE_CFG),
            ("classifier", GaussianNB()),
        ]),
        "DecisionTree": ImbPipeline([
            ("smote", SMOTE_CFG),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]),
    }

    rows = []
    for name, pipe in models_subset.items():
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        pr, rc, f1, _ = precision_recall_fscore_support(
            y_te, y_pred, average="macro", zero_division=0
        )
        rows.append({
            "model": name,
            "accuracy": acc,
            "precision": pr,
            "recall": rc,
            "f1": f1
        })
        print(f"{name:15s} -> Acc={acc:.4f} | Prec={pr:.4f} | Rec={rc:.4f} | F1={f1:.4f}")

    df_subset = pd.DataFrame(rows).set_index("model").sort_values("accuracy", ascending=False)
    print("\nRingkasan kinerja subset", label_subset)
    print(df_subset)

    # Grafik perbandingan akurasi
    plt.figure(figsize=(9, 5))
    bars = plt.bar(df_subset.index, df_subset["accuracy"])
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=25, ha="right")
    plt.title(f"Perbandingan Kinerja Model\nSubset 6 Fitur ({label_subset})")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0,
                 h + 0.01,
                 f"{h*100:.1f}%",
                 ha="center", va="bottom", fontsize=8, fontweight="bold")
    plt.tight_layout()
    fname = f"perf_subset_6feat_{label_subset.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(fname, dpi=300)
    plt.show()
    print("Grafik disimpan sebagai:", fname)

    return df_subset

# --------------------------------------------------------
# 21.1 6 Fitur Teratas Berdasarkan Mutual Information
# --------------------------------------------------------
print("\n" + "-"*20 + " 6 Fitur Terbaik - Mutual Information " + "-"*20)

mi_scores = mutual_info_classif(X_baseline, y, discrete_features='auto', random_state=42)
mi_scores = np.array(mi_scores)

mi_sorted_idx = np.argsort(mi_scores)[::-1]
mi_sorted_feats = [X_baseline.columns[i] for i in mi_sorted_idx]

print("\nNilai Mutual Information per fitur (urut dari tertinggi):")
for i in mi_sorted_idx:
    print(f"{X_baseline.columns[i]:30s} : {mi_scores[i]:.6f}")

top6_mi_features = mi_sorted_feats[:6]
print("\nTop-6 Fitur (Mutual Information):", top6_mi_features)

# Grafik MI
plt.figure(figsize=(9,4))
top_n_mi = min(15, len(mi_sorted_feats))
top_idx_plot = mi_sorted_idx[:top_n_mi]
plt.bar([X_baseline.columns[i] for i in top_idx_plot],
        mi_scores[top_idx_plot])
plt.xticks(rotation=75, ha="right")
plt.ylabel("Mutual Information")
plt.title("Mutual Information per Fitur (Top 15)")
plt.tight_layout()
plt.savefig("mutual_information_top15.png", dpi=300)
plt.show()

# Evaluasi semua model dengan 6 fitur teratas MI
df_mi_6 = evaluate_models_on_subset_6feat(top6_mi_features, "Mutual Information (Top-6)")

# --------------------------------------------------------
# 21.2 6 Fitur Teratas Berdasarkan LinearSVC L1
# --------------------------------------------------------
print("\n" + "-"*20 + " 6 Fitur Terbaik - LinearSVC L1 " + "-"*20)

lsvc_l1 = LinearSVC(penalty="l1", dual=False, max_iter=5000, random_state=42)
lsvc_l1.fit(X_baseline, y)

coef_l1 = np.mean(np.abs(lsvc_l1.coef_), axis=0)
l1_sorted_idx = np.argsort(coef_l1)[::-1]
l1_sorted_feats = [X_baseline.columns[i] for i in l1_sorted_idx]

print("\nBobot absolut LinearSVC L1 per fitur (urut dari tertinggi):")
for i in l1_sorted_idx:
    print(f"{X_baseline.columns[i]:30s} : {coef_l1[i]:.6f}")

top6_l1_features = l1_sorted_feats[:6]
print("\nTop-6 Fitur (LinearSVC L1):", top6_l1_features)

plt.figure(figsize=(9,4))
top_n_l1 = min(15, len(l1_sorted_feats))
idx_plot_l1 = l1_sorted_idx[:top_n_l1]
plt.bar([X_baseline.columns[i] for i in idx_plot_l1],
        coef_l1[idx_plot_l1])
plt.xticks(rotation=75, ha="right")
plt.ylabel("|Coef| LinearSVC L1")
plt.title("Feature Importance LinearSVC L1 (Top 15)")
plt.tight_layout()
plt.savefig("linearsvc_l1_top15.png", dpi=300)
plt.show()

df_l1_6 = evaluate_models_on_subset_6feat(top6_l1_features, "LinearSVC L1 (Top-6)")

# --------------------------------------------------------
# 21.3 6 Fitur Teratas Berdasarkan LinearSVC L2
# --------------------------------------------------------
print("\n" + "-"*20 + " 6 Fitur Terbaik - LinearSVC L2 " + "-"*20)

lsvc_l2 = LinearSVC(penalty="l2", dual=True, max_iter=5000, random_state=42)
lsvc_l2.fit(X_baseline, y)

coef_l2 = np.mean(np.abs(lsvc_l2.coef_), axis=0)
l2_sorted_idx = np.argsort(coef_l2)[::-1]
l2_sorted_feats = [X_baseline.columns[i] for i in l2_sorted_idx]

print("\nBobot absolut LinearSVC L2 per fitur (urut dari tertinggi):")
for i in l2_sorted_idx:
    print(f"{X_baseline.columns[i]:30s} : {coef_l2[i]:.6f}")

top6_l2_features = l2_sorted_feats[:6]
print("\nTop-6 Fitur (LinearSVC L2):", top6_l2_features)

plt.figure(figsize=(9,4))
top_n_l2 = min(15, len(l2_sorted_feats))
idx_plot_l2 = l2_sorted_idx[:top_n_l2]
plt.bar([X_baseline.columns[i] for i in idx_plot_l2],
        coef_l2[idx_plot_l2])
plt.xticks(rotation=75, ha="right")
plt.ylabel("|Coef| LinearSVC L2")
plt.title("Feature Importance LinearSVC L2 (Top 15)")
plt.tight_layout()
plt.savefig("linearsvc_l2_top15.png", dpi=300)
plt.show()

df_l2_6 = evaluate_models_on_subset_6feat(top6_l2_features, "LinearSVC L2 (Top-6)")

# --------------------------------------------------------
# 21.4 6 Fitur Teratas Berdasarkan SHAP
#      - SVM-linear
#      - SVM-rbf
#      - DecisionTree
# --------------------------------------------------------
print("\n" + "-"*20 + " 6 Fitur Terbaik - SHAP " + "-"*20)

feature_names_shap = np.array(fitur_shap)

def get_shap_ranking_all(model_key):
    shap_imp = shap_importance_dict[model_key]
    idx_sorted = np.argsort(shap_imp)[::-1]
    all_feats_sorted = list(feature_names_shap[idx_sorted])
    return all_feats_sorted, shap_imp, idx_sorted

# ---------- a) SVM-linear ----------
all_linear_features, shap_linear, linear_idx_sorted = get_shap_ranking_all("SVM-linear")

print("\n=== Ranking Mean |SHAP| - SVM-linear (SEMUA FITUR) ===")
for rank, idx_f in enumerate(linear_idx_sorted, start=1):
    print(f"{rank:2d}. {feature_names_shap[idx_f]} -> mean|SHAP|={shap_linear[idx_f]:.6f}")
plt.figure(figsize=(8, 4))
plt.bar(feature_names_shap[linear_idx_sorted], shap_linear[linear_idx_sorted])
plt.xticks(rotation=75, ha="right")
plt.ylabel("Mean |SHAP|")
plt.title("Kernel SHAP Feature Importance - SVM-linear (SEMUA FITUR)")
plt.tight_layout()
plt.savefig("shap_linear_allfeatures.png", dpi=300)
plt.show()

top6_linear_features = all_linear_features[:6]
print("\nTop-6 Fitur SHAP - SVM-linear:", top6_linear_features)
df_shap_linear_6 = evaluate_models_on_subset_6feat(top6_linear_features, "SHAP SVM-linear (Top-6)")


# ---------- b) SVM-rbf ----------
all_rbf_features, shap_rbf, rbf_idx_sorted = get_shap_ranking_all("SVM-RBF")

print("\n=== Ranking Mean |SHAP| - SVM-RBF (SEMUA FITUR) ===")
for rank, idx_f in enumerate(rbf_idx_sorted, start=1):
    print(f"{rank:2d}. {feature_names_shap[idx_f]} -> mean|SHAP|={shap_rbf[idx_f]:.6f}")
plt.figure(figsize=(8, 4))
plt.bar(feature_names_shap[rbf_idx_sorted], shap_rbf[rbf_idx_sorted])
plt.xticks(rotation=75, ha="right")
plt.ylabel("Mean |SHAP|")
plt.title("Kernel SHAP Feature Importance - SVM-RBF (SEMUA FITUR)")
plt.tight_layout()
plt.savefig("shap_rbf_allfeatures.png", dpi=300)
plt.show()

top6_rbf_features = all_rbf_features[:6]
print("\nTop-6 Fitur SHAP - SVM-RBF:", top6_rbf_features)
df_shap_rbf_6 = evaluate_models_on_subset_6feat(top6_rbf_features, "SHAP SVM-RBF (Top-6)")


# ---------- c) DecisionTree ----------
all_dt_features, shap_dt, dt_idx_sorted = get_shap_ranking_all("DecisionTree")

print("\n=== Ranking Mean |SHAP| - DecisionTree (SEMUA FITUR) ===")
for rank, idx_f in enumerate(dt_idx_sorted, start=1):
    print(f"{rank:2d}. {feature_names_shap[idx_f]} -> mean|SHAP|={shap_dt[idx_f]:.6f}")

plt.figure(figsize=(8, 4))
plt.bar(feature_names_shap[dt_idx_sorted], shap_dt[dt_idx_sorted])
plt.xticks(rotation=75, ha="right")
plt.ylabel("Mean |SHAP|")
plt.title("Tree SHAP Feature Importance - DecisionTree (SEMUA FITUR)")
plt.tight_layout()
plt.savefig("shap_decisiontree_allfeatures.png", dpi=300)
plt.show()

top6_dt_features = all_dt_features[:6]
print("\nTop-6 Fitur SHAP - DecisionTree:", top6_dt_features)
df_shap_dt_6 = evaluate_models_on_subset_6feat(top6_dt_features, "SHAP DecisionTree (Top-6)")
print("\n Analisis 6 fitur terpiliih selesai.")

# ========================================================
# 22. PERBANDINGAN NILAI k PADA K-FOLD
#     (k = 5, 10, 15, 30) UNTUK SEMUA MODEL - SEMUA FITUR
# ========================================================

print("\n" + "="*25 + " PERBANDINGAN NILAI k PADA K-FOLD (SEMUA MODEL - ALL FEATURES) " + "="*25)

k_values = [5, 10, 15, 30]
multi_k_results_all = []

for k in k_values:
    print(f"\n--- Evaluasi K-Fold dengan k = {k} ---")
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


    for name, pipe in model_dict_all.items():
        # Baseline (all) pakai X_baseline (sudah di-LabelEncoder, semua numerik)
        if "Baseline (all)" in name:
            X_used = X_baseline
            y_used = y
        else:
            # Model lain sudah pakai preprocessor masing-masing (ColumnTransformer)
            X_used = X
            y_used = y

        scores = cross_val_score(pipe, X_used, y_used, cv=kf, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()

        multi_k_results_all.append({
            "model": name,
            "k": k,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc
        })

        print(f"Model: {name:20s} | k={k:2d} | mean={mean_acc:.4f} ± {std_acc:.4f} "
              f"({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")

# Jadikan DataFrame
df_multi_k_all = pd.DataFrame(multi_k_results_all)
print("\n=== Ringkasan K-Fold Multi-k (Semua Model - All Features) ===")
print(df_multi_k_all)

# Pivot: baris = k, kolom = model, nilai = mean_accuracy
pivot_multi_k_all = df_multi_k_all.pivot(index="k", columns="model", values="mean_accuracy")
print("\n=== Pivot Mean Accuracy (index = k, columns = model) ===")
print(pivot_multi_k_all)

# Simpan ke Excel/CSV untuk laporan
df_multi_k_all.to_csv("kfold_multi_k_semua_model_allfitur.csv", index=False)
pivot_multi_k_all.to_excel("kfold_multi_k_semua_model_allfitur_pivot.xlsx")

# --------------------------------------------------------
# Grafik: garis per model, sumbu-x = k, sumbu-y = mean accuracy
# --------------------------------------------------------
plt.figure(figsize=(10, 6))
for model_name in pivot_multi_k_all.columns:
    plt.plot(pivot_multi_k_all.index,
             pivot_multi_k_all[model_name],
             marker="o",
             linewidth=2,
             label=model_name)

plt.xlabel("Jumlah Fold (k)")
plt.ylabel("Mean Accuracy")
plt.ylim(0, 1.05)
plt.title("Perbandingan Mean Accuracy K-Fold\n(k = 5, 10, 15, 30) – Semua Model (All Features)")
plt.xticks(list(k_values))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("kfold_multi_k_all_models_all_features.png", dpi=300)
plt.show()

print("\nFile multi-k tersimpan sebagai:")
print("- kfold_multi_k_semua_model_allfitur.csv")
print("- kfold_multi_k_semua_model_allfitur_pivot.xlsx")
print("- kfold_multi_k_all_models_all_features.png")

# ========================================================
# 22b. PERBANDINGAN NILAI k PADA K-FOLD
#      (k = 5, 10, 15, 30) UNTUK TOP-3 MODEL (6 FITUR TERPILIH)
# ========================================================

print("\n" + "="*25 + " PERBANDINGAN NILAI k (TOP-3 MODEL - 6 FITUR TERPILIH) " + "="*25)

# Pastikan df_6fitur & top3_sel sudah ada dari bagian 19b
mask_sel = df_all_sorted["model"].str.contains("6fitur", case=False)
df_6fitur = df_all_sorted[mask_sel].copy()
top_n = min(3, len(df_6fitur))
top3_sel = df_6fitur.head(top_n).reset_index(drop=True)

print(f"\n=== {top_n} Model Terbaik (Kategori 6 Fitur Terpilih) ===")
print(top3_sel[["model", "mean_score"]])

# Data khusus 3 fitur terpilih
X_sel_all = X_baseline[selected_features]
y_all = y

k_values = [5, 10, 15, 30]
multi_k_results_sel = []

for k in k_values:
    print(f"\n--- Evaluasi K-Fold dengan k = {k} (3 fitur terpilih) ---")
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


    for _, row in top3_sel.iterrows():
        model_name_full = row["model"]   # contoh: "SVM-rbf (6fitur)"
        low_name = model_name_full.lower()

        # Bangun pipeline sesuai nama model (sama pola seperti 19b)
        if "svm" in low_name:
            if "rbf" in low_name:
                kernel = "rbf"
            elif "poly" in low_name:
                kernel = "poly"
            elif "sigmoid" in low_name:
                kernel = "sigmoid"
            else:
                kernel = "linear"

            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", SVC(kernel=kernel, probability=True, random_state=42))
            ])

        elif "knn" in low_name:
            pipe = ImbPipeline([
                ("scaler", StandardScaler()),
                ("smote", SMOTE_CFG),
                ("classifier", KNeighborsClassifier(n_neighbors=5))
            ])

        elif "naive" in low_name:
            pipe = ImbPipeline([
                ("smote", SMOTE_CFG),
                ("classifier", GaussianNB())
            ])

        elif "decision" in low_name:
            pipe = ImbPipeline([
                ("smote", SMOTE_CFG),
                ("classifier", DecisionTreeClassifier(random_state=42))
            ])

        elif "baseline" in low_name:
            pipe = Pipeline([
                ("classifier", SVC(kernel="linear", probability=True, random_state=42))
            ])

        else:
            print(f"Model '{model_name_full}' tidak dikenali → dilewati.")
            continue

        scores = cross_val_score(pipe, X_sel_all, y_all, cv=kf, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()

        multi_k_results_sel.append({
            "model": model_name_full,
            "k": k,
            "mean_accuracy": mean_acc,
            "std_accuracy": std_acc
        })

        print(f"Model: {model_name_full:25s} | k={k:2d} | mean={mean_acc:.4f} ± {std_acc:.4f} "
              f"({mean_acc*100:.2f}% ± {std_acc*100:.2f}%)")

# DataFrame & pivot
df_multi_k_sel = pd.DataFrame(multi_k_results_sel)
print("\n=== Ringkasan K-Fold Multi-k (Top-3 Model - 3 Fitur) ===")
print(df_multi_k_sel)

pivot_multi_k_sel = df_multi_k_sel.pivot(index="k", columns="model", values="mean_accuracy")
print("\n=== Pivot Mean Accuracy (Top-3 Model - 3 Fitur) ===")
print(pivot_multi_k_sel)

df_multi_k_sel.to_csv("kfold_multi_k_top3model_6fitur.csv", index=False)
pivot_multi_k_sel.to_excel("kfold_multi_k_top3model_6fitur_pivot.xlsx")

# Grafik garis per model
plt.figure(figsize=(8, 5))
for model_name in pivot_multi_k_sel.columns:
    plt.plot(pivot_multi_k_sel.index,
             pivot_multi_k_sel[model_name],
             marker="o",
             linewidth=2,
             label=model_name)

plt.xlabel("Jumlah Fold (k)")
plt.ylabel("Mean Accuracy (3 Fitur)")
plt.ylim(0, 1.05)
plt.title("Perbandingan Mean Accuracy K-Fold\nTop-3 Model (3 Fitur Terpilih) – k = 5, 10, 15, 30")
plt.xticks(list(k_values))
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("kfold_multi_k_top3model_6fitur.png", dpi=300)
plt.show()

print("\nFile multi-k 6 fitur tersimpan sebagai:")
print("- kfold_multi_k_top3model_6fitur.csv")
print("- kfold_multi_k_top3model_6fitur_pivot.xlsx")
print("- kfold_multi_k_top3model_6fitur.png")

# =========================================
# Helper: pipeline sesuai nama model
# (harus sama dengan evaluate_models_on_subset_6feat)
# =========================================
def build_pipeline_by_name(model_name: str):
    name = model_name.lower()

    if "baseline" in name:
        return Pipeline([("classifier", SVC(kernel="linear", probability=True, random_state=42))])

    if "svm-linear" in name or (("svm" in name) and ("linear" in name)):
        return ImbPipeline([
            ("smote", SMOTE_CFG),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="linear", probability=True, random_state=42))
        ])

    if "svm-rbf" in name or (("svm" in name) and ("rbf" in name)):
        return ImbPipeline([
            ("smote", SMOTE_CFG),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
        ])

    if "svm-poly" in name or (("svm" in name) and ("poly" in name)):
        return ImbPipeline([
            ("smote", SMOTE_CFG),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="poly", probability=True, random_state=42))
        ])

    if "svm-sigmoid" in name or (("svm" in name) and ("sigmoid" in name)):
        return ImbPipeline([
            ("smote", SMOTE_CFG),
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="sigmoid", probability=True, random_state=42))
        ])

    if "knn" in name:
        return ImbPipeline([
            ("smote", SMOTE_CFG),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ])

    if "naive" in name or "bayes" in name:
        return ImbPipeline([("smote", SMOTE_CFG), ("classifier", GaussianNB())])

    if "decision" in name or "tree" in name:
        return ImbPipeline([("smote", SMOTE_CFG), ("classifier", DecisionTreeClassifier(random_state=42))])

    raise ValueError(f"Model tidak dikenali: {model_name}")


# =========================================
# Helper: 5-Fold CV untuk Top-3 model terbaik pada subset fitur
# =========================================
def kfold_top3_for_subset(selected_features, label_subset, k=5):
    # 1) cari top-3 model terbaik (berdasarkan accuracy di holdout split)
    df_subset = evaluate_models_on_subset_6feat(selected_features, label_subset)
    top3_models = df_subset.head(3).index.tolist()

    # 2) jalankan CV untuk top-3 itu
    X_sub = X_baseline[selected_features]
    y_sub = y

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


    rows = []
    for m in top3_models:
        pipe = build_pipeline_by_name(m)
        scores = cross_val_score(pipe, X_sub, y_sub, cv=kf, scoring="accuracy")
        rows.append({
            "subset": label_subset,
            "model": m,
            "mean_accuracy": scores.mean(),
            "std_accuracy": scores.std()
        })

    df_k = pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)
    return df_k


# =========================================
# Helper: plot 1 panel (mean & std)
# =========================================
def plot_kfold_panel(ax, df_k, title):
    models = df_k["model"].tolist()
    mean_acc = df_k["mean_accuracy"].values
    std_acc  = df_k["std_accuracy"].values

    x = np.arange(len(models))
    w = 0.35

    ax.bar(x - w/2, mean_acc, width=w, label="Mean of Accuracy")
    ax.bar(x + w/2, std_acc,  width=w, label="Std. Accuracy")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=9, weight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # angka kecil di atas batang
    for i, v in enumerate(mean_acc):
        ax.text(i - w/2, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
    for i, v in enumerate(std_acc):
        ax.text(i + w/2, v + 0.02, f"{v:.4f}", ha="center", va="bottom", fontsize=7)


# =========================================
# 7 skenario yang kamu minta
# (Pastikan list fitur ini sudah ada)
# =========================================
subsets = [
    ("All Features (semua fitur)", list(X_baseline.columns)),       # kalau mau "ALL FEATURES" beneran, pakai X_baseline.columns
    ("Selected by LinearSVC L1 (Top-6)", top6_l1_features),
    ("Selected by LinearSVC L2 (Top-6)", top6_l2_features),
    ("Selected by Mutual Information (Top-6)", top6_mi_features),
    ("Selected by SHAP Decision Tree (Top-6)", top6_dt_features),
    ("Selected by SHAP SVM Kernel Linear (Top-6)", top6_linear_features),
    ("Selected by SHAP SVM Kernel RBF (Top-6)", top6_rbf_features),
]

all_tables = []
for title, feats in subsets:
    dfk = kfold_top3_for_subset(feats, title, k=5)
    all_tables.append(dfk)

df_all_kfold = pd.concat(all_tables, ignore_index=True)
df_all_kfold.to_csv("kfold_top3_tiap_subset_7metode.csv", index=False)
print("\n csv tersimpan: kfold_top3_tiap_subset_7metode.csv")
print(df_all_kfold)

# Plot 7 panel (layout 4 + 3)
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
axes = axes.flatten()

for i, (title, feats) in enumerate(subsets):
    dfk = df_all_kfold[df_all_kfold["subset"] == title].copy()
    plot_kfold_panel(axes[i], dfk, title)

if len(subsets) < len(axes):
    axes[-1].axis("off")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("kfold_7metode_top3_panel.png", dpi=300)
plt.show()

print("\n Gambar panel tersimpan dengan nama kfold_7metode_top3_panel.png")