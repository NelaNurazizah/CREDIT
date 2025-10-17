# ============================================================
# SISTEM PREDIKSI KELAYAKAN KREDIT
# Logistic Regression + Flask Web App
# ============================================================

# -------- Import library --------
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template, request
import os

# ============================================================
# BAGIAN 1. PELATIHAN MODEL
# ============================================================

print("=== Memulai pelatihan model ===")

# 1. Load dataset
df = pd.read_csv('credit_default.csv')
target_col = 'default.payment.next.month'

# 2. Tangani missing value
df = df.fillna(df.median(numeric_only=True))

# 3. Pilih fitur utama
features = ['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3',
            'SEX','MARRIAGE','EDUCATION']

X = df[features]
y = df[target_col]

# 4. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Latih model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 7. Simpan model dan scaler
joblib.dump(model, 'model_credit.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(features, 'feature_names.pkl')

print("Model dan scaler berhasil disimpan.")
print("=== Pelatihan selesai ===\n")

# ============================================================
# BAGIAN 2. APLIKASI FLASK
# ============================================================

app = Flask(__name__)

# Load model dan scaler
model = joblib.load('model_credit.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Halaman utama (form input)
@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>Prediksi Risiko Kartu Kredit</title>
        <style>
            body {font-family: Arial; background: #f8f8f8; padding: 20px;}
            h2 {color: #2e5b85;}
            form {background: white; padding: 20px; border-radius: 10px; width: 320px;}
            input, select {width: 100%; margin-bottom: 10px; padding: 8px;}
            button {width: 100%; padding: 10px; background: #2e5b85; color: white; border: none; cursor: pointer;}
            button:hover {background: #447ebd;}
        </style>
    </head>
    <body>
        <h2>Prediksi Risiko Gagal Bayar</h2>
        <form action="/predict" method="post">
            <label>Batas Kredit (LIMIT_BAL)</label>
            <input type="number" name="LIMIT_BAL" required>

            <label>Usia (AGE)</label>
            <input type="number" name="AGE" required>

            <label>Tagihan Bulan ke-1 (BILL_AMT1)</label>
            <input type="number" name="BILL_AMT1" required>

            <label>Tagihan Bulan ke-2 (BILL_AMT2)</label>
            <input type="number" name="BILL_AMT2" required>

            <label>Tagihan Bulan ke-3 (BILL_AMT3)</label>
            <input type="number" name="BILL_AMT3" required>

            <label>Jenis Kelamin (SEX)</label>
            <select name="SEX" required>
                <option value="1">Laki-laki</option>
                <option value="2">Perempuan</option>
            </select>

            <label>Status Pernikahan (MARRIAGE)</label>
            <select name="MARRIAGE" required>
                <option value="1">Menikah</option>
                <option value="2">Lajang</option>
                <option value="3">Lainnya</option>
            </select>

            <label>Pendidikan (EDUCATION)</label>
            <select name="EDUCATION" required>
                <option value="1">S2/S3</option>
                <option value="2">S1</option>
                <option value="3">SMA</option>
                <option value="4">Lainnya</option>
            </select>

            <button type="submit">Prediksi</button>
        </form>
    </body>
    </html>
    '''

# Halaman hasil prediksi
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil input dari form
        LIMIT_BAL = float(request.form['LIMIT_BAL'])
        AGE = float(request.form['AGE'])
        BILL_AMT1 = float(request.form['BILL_AMT1'])
        BILL_AMT2 = float(request.form['BILL_AMT2'])
        BILL_AMT3 = float(request.form['BILL_AMT3'])
        SEX = float(request.form['SEX'])
        MARRIAGE = float(request.form['MARRIAGE'])
        EDUCATION = float(request.form['EDUCATION'])

        # Susun data ke dalam urutan fitur
        input_data = np.array([[LIMIT_BAL, AGE, BILL_AMT1, BILL_AMT2, BILL_AMT3,
                                SEX, MARRIAGE, EDUCATION]])

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediksi
        prob = model.predict_proba(input_scaled)[0, 1]
        pred_label = 'Berisiko Gagal Bayar' if prob >= 0.5 else 'Tidak Berisiko'

        # Tampilkan hasil dalam halaman sederhana
        return f'''
        <html>
        <head>
            <title>Hasil Prediksi</title>
            <style>
                body {{font-family: Arial; background: #f8f8f8; padding: 20px;}}
                .box {{background: white; padding: 20px; border-radius: 10px; width: 320px;}}
                h2 {{color: #2e5b85;}}
                .risk {{color: #c0392b; font-weight: bold;}}
                .safe {{color: #27ae60; font-weight: bold;}}
                a {{text-decoration: none; color: #2e5b85;}}
            </style>
        </head>
        <body>
            <div class="box">
                <h2>Hasil Prediksi</h2>
                <p class="{ 'risk' if pred_label=='Berisiko Gagal Bayar' else 'safe' }">{pred_label}</p>
                <p>Probabilitas: {prob*100:.2f}%</p>
                <a href="/">Kembali</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Terjadi kesalahan: {e}"

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
