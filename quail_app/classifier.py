import cv2
import numpy as np
import os
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Ekstraksi fitur RGB ---
def normalize_rgb(image):
    B, G, R = cv2.split(image)
    total = R + G + B + 1e-8  # Hindari pembagian nol
    r, g, b = (R / total).mean(), (G / total).mean(), (B / total).mean()
    return [r, g, b]

# --- Ekstraksi fitur tekstur ---
def extract_texture(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(thresh == 255) / (338 * 338)
    return [white_ratio]

# --- Gabungkan semua fitur ---
def extract_features(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Gambar tidak ditemukan: {path}")
    image = cv2.resize(image, (338, 338))
    rgb = normalize_rgb(image)
    texture = extract_texture(image)
    return rgb + texture

# --- Fungsi training dan simpan model ---
def train_model(folder='media/dataset_telur_puyuh'):
    X, y = [], []
    for subdir, _, files in os.walk(folder):  # ✅ Rekursif ke subfolder
        for file in files:
            if file.lower().endswith(('.jpg', '.png')):
                label = os.path.basename(subdir)  # ✅ Label = nama folder
                path = os.path.join(subdir, file)
                try:
                    feat = extract_features(path)
                    X.append(feat)
                    y.append(label)
                except Exception as e:
                    print(f"[WARNING] Gagal ekstrak fitur dari {file}: {e}")

    if not X:
        print("[ERROR] Tidak ada data yang berhasil diproses.")
        return

    # Encode label
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Scaling fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KNN
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X_scaled, y_encoded)

    # Simpan model
    joblib.dump(clf, 'knn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'label_encoder.pkl')
    print("[INFO] Model, scaler, dan encoder berhasil disimpan.")

    # Encode label
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Scaling fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train KNN
    clf = KNeighborsClassifier(n_neighbors=7)
    clf.fit(X_scaled, y_encoded)

    # Simpan model
    joblib.dump(clf, 'knn_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'label_encoder.pkl')
    print("[INFO] Model, scaler, dan encoder berhasil disimpan.")

# --- Fungsi prediksi gambar baru ---
def predict_image(image_path):
    try:
        clf = joblib.load('knn_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('label_encoder.pkl')
    except Exception as e:
        print(f"[DEBUG] Model atau scaler tidak ditemukan: {e}")
        return None

    try:
        feat = extract_features(image_path)
        scaled = scaler.transform([feat])
        pred = clf.predict(scaled)
        label = encoder.inverse_transform(pred)
        return label[0]
    except Exception as e:
        print(f"[ERROR] Gagal memproses gambar: {e}")
        return None

# --- Program utama ---
if __name__ == "__main__":
    print("== Mulai training model dari dataset ==")
    train_model()

    print("\n== Mulai prediksi gambar uji ==")
    test_img = 'media/dataset_telur_puyuh/Data_Testing/Baik/baik_1.jpg'  # Ganti dengan path gambar uji kamu
    if os.path.exists(test_img):
        hasil = predict_image(test_img)
        print("Hasil prediksi:", hasil)
    else:
        print(f"[WARNING] Gambar uji tidak ditemukan: {test_img}")
