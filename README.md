# Saudi Used Cars Price Prediction (Capstone Project Module 3)

Project ini membangun model Machine Learning untuk **memprediksi harga mobil bekas (Price)** di Arab Saudi berdasarkan informasi kendaraan pada listing. Dataset diambil dari **syarah.com**.

## Objective
- Memprediksi **Price** (harga mobil bekas) menggunakan fitur kendaraan.
- Membandingkan beberapa model regresi dan memilih model terbaik berdasarkan **5-Fold Cross Validation (RMSE)**.
- Mengevaluasi performa final pada **test set** dan melakukan visualisasi hasil prediksi.

---

## Dataset
**Source:** syarah.com (Saudi used cars listings)  
**Target (y):** `Price`  
**Features (X):**
- Categorical: `Type`, `Region`, `Make`, `Gear_Type`, `Origin`, `Options`
- Numeric: `Year`, `Engine_Size`, `Mileage`
- Boolean: `Negotiable`

> Catatan penting: Banyak baris memiliki `Price = 0` yang mengindikasikan harga negotiable/tidak tercantum sehingga tidak merepresentasikan harga sebenarnya.

---

## Project Workflow

### 1) Data Understanding
- Mengecek struktur data (`info`, `dtypes`) dan memastikan kesiapan untuk pemodelan.
- Identifikasi tipe fitur: kategorikal, numerik, dan boolean.

### 2) Data Cleaning
- Menghapus data dengan `Price = 0` karena tidak merepresentasikan harga sebenarnya (negotiable / missing price).

### 3) Train-Test Split
- Split data: **80% train / 20% test**
- `random_state = 42`

### 4) Preprocessing (Pipeline + ColumnTransformer)
Tujuan: memastikan preprocessing konsisten di train/test dan mencegah **data leakage**.

- **Numerik (+ boolean)**: `Year`, `Engine_Size`, `Mileage`, `Negotiable`
  - Imputer: `median`
  - Scaler: `StandardScaler`
- **Kategorikal**: `Type`, `Region`, `Make`, `Gear_Type`, `Origin`, `Options`
  - Imputer: `most_frequent`
  - Encoder: `OneHotEncoder(handle_unknown="ignore")`
  - Output dibuat dense (`sparse_output=False` / fallback `sparse=False`) agar kompatibel untuk evaluasi tertentu.

### 5) Baseline Model
- Baseline sederhana: memprediksi semua data test dengan **median `y_train`**.
- Digunakan untuk memastikan model ML memberikan improvement.

### 6) Model Benchmarking + Cross Validation
Model yang dibandingkan (semua memakai preprocessing pipeline yang sama):
- Linear Regression
- Random Forest Regressor
- HistGradientBoosting Regressor

Evaluasi model:
- **5-Fold Cross Validation**
- Metrik: **RMSE** (`neg_root_mean_squared_error`)

**Hasil CV (RMSE mean ± std):**
- Random Forest: **32,886.78 ± 3,537.72** (terbaik)
- HistGradientBoosting: 33,131.85 ± 3,943.74
- Linear Regression: 45,015.48 ± 6,056.17

### 7) Final Model Evaluation (Test Set)
Model terbaik (berdasarkan CV RMSE terendah): **Random Forest**

Hasil evaluasi test set:
- **MAE:** 17,741.83
- **RMSE:** 36,810.35
- **R²:** 0.7434

### 8) Visualization
- Scatter plot **Actual vs Predicted**
- Histogram **Residual Distribution**

---

## Results Summary
- Model terbaik untuk dataset ini adalah **Random Forest Regressor**.
- Model mampu menjelaskan ~**74% variasi harga** (R² ≈ 0.74) pada test set.
- Pipeline preprocessing membantu menghindari data leakage dan membuat eksperimen lebih robust.

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib

---

## Repository Structure (suggested)
