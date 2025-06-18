import os
import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from django.conf import settings
import tempfile
import joblib

label_map = {'Baik': 1, 'Sedang': 2, 'Buruk': 3}
label_map_reverse = {v: k for k, v in label_map.items()}

# ========================
# Ekstraksi fitur dari gambar (RGB + Otsu + fusi)
# ========================
def extract_features(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Gambar tidak dapat dibaca: {image_path}")
            return None

        img = cv2.resize(img, (256, 256))

        # Rata-rata nilai RGB
        R = np.mean(img[:, :, 2])
        G = np.mean(img[:, :, 1])
        B = np.mean(img[:, :, 0])

        # Otsu
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        black = np.sum(otsu == 0)
        otsu_mean = np.mean(otsu)

        # Fusi: kombinasi warna dan bentuk (sederhana)
        fusi = int((R + G + B) / 3 + black / 1000)

        fitur = np.array([R, G, B, otsu_mean, black, fusi], dtype=np.float32)
        return fitur

    except Exception as e:
        print(f"[ERROR extract_features]: {e}")
        return None

# ========================
# Load data pelatihan
# ========================
def load_data(train_path):
    X, y = [], []
    for kategori in ['Baik', 'Sedang', 'Buruk']:
        folder = os.path.join(train_path, kategori)
        if not os.path.exists(folder):
            continue
        for img_name in sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]):
            path = os.path.join(folder, img_name)
            fitur = extract_features(path)
            if fitur is not None:
                X.append(fitur)
                y.append(label_map[kategori])
    return X, y

# ========================
# Fungsi pelatihan dan simpan model
# ========================
def train_model():
    train_path = os.path.join(settings.MEDIA_ROOT, 'dataset_telur_puyuh', 'Data_Training')
    X_train, y_train = load_data(train_path)

    if not X_train or not y_train:
        return "Data training kosong!"

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_scaled, y_train)

    with open(os.path.join(settings.BASE_DIR, 'knn_model.pkl'), 'wb') as f:
        pickle.dump(knn, f)
    with open(os.path.join(settings.BASE_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)

    return "Model dan scaler berhasil disimpan."

# ========================
# Fungsi pengujian banyak gambar
# ========================
def test_model():
    base_path = os.path.join(settings.MEDIA_ROOT, 'dataset_telur_puyuh')
    train_path = os.path.join(base_path, 'Data_Training')
    test_path = os.path.join(base_path, 'Data_Testing')

    gray_out = os.path.join(base_path, 'Data_Testing_Gray')
    otsu_out = os.path.join(base_path, 'Data_Testing_Otsu')
    os.makedirs(gray_out, exist_ok=True)
    os.makedirs(otsu_out, exist_ok=True)

    X_train, y_train = load_data(train_path)
    if not X_train or not y_train:
        print("[ERROR] Data training kosong.")
        return []

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train_scaled, y_train)

    test_data = []
    for kategori in ['Baik', 'Sedang', 'Buruk']:
        folder = os.path.join(test_path, kategori)
        if not os.path.exists(folder):
            continue

        gray_folder = os.path.join(gray_out, kategori)
        otsu_folder = os.path.join(otsu_out, kategori)
        os.makedirs(gray_folder, exist_ok=True)
        os.makedirs(otsu_folder, exist_ok=True)

        for img_name in sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])[:4]:
            full_path = os.path.join(folder, img_name)
            fitur = extract_features(full_path)
            if fitur is None:
                continue

            try:
                img = cv2.imread(full_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cv2.imwrite(os.path.join(gray_folder, img_name), gray)

                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(otsu_folder, img_name), otsu)

                fitur_scaled = scaler.transform([fitur])
                pred = knn.predict(fitur_scaled)[0]

                print(f"[DEBUG] File: {img_name} | Actual: {kategori} | Predicted: {label_map_reverse.get(pred, '?')}")

                test_data.append({
                    'filename': img_name,
                    'actual': kategori,
                    'fitur_array': fitur,
                    'prediksi': pred,
                    'image_url': f'{settings.MEDIA_URL}dataset_telur_puyuh/Data_Testing/{kategori}/{img_name}',
                    'grayscale_url': f'{settings.MEDIA_URL}dataset_telur_puyuh/Data_Testing_Gray/{kategori}/{img_name}',
                    'otsu_url': f'{settings.MEDIA_URL}dataset_telur_puyuh/Data_Testing_Otsu/{kategori}/{img_name}',
                })
            except Exception as e:
                print(f"[ERROR test_model prediction] {full_path}: {e}")
                continue

    return test_data

# ========================
# Fungsi klasifikasi 1 gambar upload
# ========================
def predict_image(image_path):
    try:
        fitur = extract_features(image_path)
        if fitur is None:
            print("[DEBUG] Ekstraksi fitur gagal.")
            return "Tidak dikenali"

        model_path = os.path.join(settings.BASE_DIR, 'knn_model.pkl')
        scaler_path = os.path.join(settings.BASE_DIR, 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("[DEBUG] Model atau scaler tidak ditemukan.")
            return "Model belum dilatih"

        with open(model_path, 'rb') as f:
            knn = pickle.load(f)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

        fitur_scaled = scaler.transform([fitur])
        pred = knn.predict(fitur_scaled)[0]
        label = label_map_reverse.get(pred, "Tidak diketahui")

        print(f"[DEBUG] Prediksi berhasil, hasil: {label}")
        return label

    except Exception as e:
        print(f"[ERROR predict_image] {e}")
        return "Terjadi kesalahan"

