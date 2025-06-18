from django.shortcuts import render, redirect
from django.conf import settings
import os
import uuid
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
from .model_knn import predict_image
from .classifier import predict_image, extract_features as hitung_fitur

# ========================
# View halaman utama
# ========================
def home(request):
    return render(request, 'index.html')

# ========================
# Hitung fitur gambar (RGB, Otsu, Fusi)
# ========================
def hitung_fitur(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (338, 338))

    R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
    sum_rgb = R + G + B + 1e-6
    R_norm = R / sum_rgb
    G_norm = G / sum_rgb
    B_norm = B / sum_rgb

    ratar = np.mean(R_norm)
    ratag = np.mean(G_norm)
    ratab = np.mean(B_norm)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    black_ratio = np.sum(otsu == 0) / otsu.size

    contours, _ = cv2.findContours(255 - otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fusi_value = len(contours)

    return {
        'ratar': round(ratar, 4),
        'ratag': round(ratag, 4),
        'ratab': round(ratab, 4),
        'black_ratio': round(black_ratio, 4),
        'fusi': fusi_value
    }

# ========================
# View untuk menampilkan dataset
# ========================
def dataset_view(request):
    kategori = request.GET.get('kategori', 'Baik')
    folder_path = os.path.join(settings.MEDIA_ROOT, 'dataset_telur_puyuh', kategori)
    gambar_list = []

    if not os.path.exists(folder_path):
        return render(request, 'dataset.html', {
            'kategori': kategori.capitalize(),
            'data_gambar': [],
            'error': f'Folder untuk kategori {kategori} tidak ditemukan.'
        })

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder_path, filename)
            fitur = ekstrak_fitur_dataset(path)

            gambar_list.append({
                'url': settings.MEDIA_URL + f'dataset_telur_puyuh/{kategori}/{filename}',
                **fitur
            })

    context = {
        'kategori': kategori.capitalize(),
        'data_gambar': gambar_list,
    }
    return render(request, 'dataset.html', context)

# ========================
# Upload gambar ke dataset
# ========================
def upload_dataset(request):
    if request.method == 'POST' and request.FILES.get('image'):
        kategori = request.POST.get('kategori', 'Baik')
        folder_path = os.path.join(settings.MEDIA_ROOT, 'dataset_telur_puyuh', kategori)
        os.makedirs(folder_path, exist_ok=True)

        image = request.FILES['image']
        filename = image.name
        path = os.path.join(folder_path, filename)

        with open(path, 'wb+') as f:
            for chunk in image.chunks():
                f.write(chunk)

        return redirect('dataset')

    return render(request, 'upload_dataset.html')

# ========================
# Hitung metrik dari hasil testing
# ========================
def calculate_metrics(data_hasil, y_pred, y_true):
    labels = ["Baik", "Sedang", "Buruk"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    metrics = []

    for i, label in enumerate(labels):
        TP = cm[i][i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = total - (TP + FP + FN)
        acc = ((TP + TN) / total) * 100 if total > 0 else 0

        metrics.append({
            "kelas": label,
            "TP": int(TP),
            "TN": int(TN),
            "FP": int(FP),
            "FN": int(FN),
            "akurasi": round(acc, 1)
        })

    avg_acc = round(sum([m['akurasi'] for m in metrics]) / len(metrics), 1)
    return cm, metrics, avg_acc

# ========================
# View untuk pengujian model
# ========================
def testing_view(request):
    # Data hasil klasifikasi
    data_hasil = [
        {"filename": "baik_1.jpg", "actual": "Baik", "prediksi": "Baik"},
        {"filename": "baik_2.jpg", "actual": "Baik", "prediksi": "Baik"},
        {"filename": "baik_3.jpg", "actual": "Baik", "prediksi": "Baik"},
        {"filename": "baik_4.jpg", "actual": "Baik", "prediksi": "Sedang"},
        {"filename": "sedang_1.jpg", "actual": "Sedang", "prediksi": "Baik"},
        {"filename": "sedang_2.jpg", "actual": "Sedang", "prediksi": "Sedang"},
        {"filename": "sedang_3.jpg", "actual": "Sedang", "prediksi": "Sedang"},
        {"filename": "sedang_4.jpg", "actual": "Sedang", "prediksi": "Buruk"},
        {"filename": "buruk_1.jpg", "actual": "Buruk", "prediksi": "Sedang"},
        {"filename": "buruk_2.jpg", "actual": "Buruk", "prediksi": "Buruk"},
        {"filename": "buruk_3.jpg", "actual": "Buruk", "prediksi": "Buruk"},
        {"filename": "buruk_4.jpg", "actual": "Buruk", "prediksi": "Buruk"}
    ]

    labels = ['Baik', 'Sedang', 'Buruk']

    # Metrik per kelas (data statis)
    metrics_per_class = {
        'Baik': {"TP": 3, "TN": 7, "FP": 1, "FN": 1, "accuracy": 83.33},
        'Sedang': {"TP": 2, "TN": 6, "FP": 2, "FN": 2, "accuracy": 66.67},
        'Buruk': {"TP": 3, "TN": 7, "FP": 1, "FN": 1, "accuracy": 83.33}
    }

    overall_accuracy = 66.67 
    #rumus overall_accuracy = total keseluruhan data testing TP / total data testing keseluruhan = 8/12 = 2/3 = 0.6667 = 66.67%

    # Confusion matrix 
    confusion_matrix_telur = {
        'Baik': [3, 1, 0],
        'Sedang': [1, 2, 1],
        'Buruk': [0, 1, 3]
    }

    # Tambahkan info benar/salah dan URL gambar
    for item in data_hasil:
        item['benar'] = item['actual'] == item['prediksi']
        kelas = item['actual']
        fname = item['filename']
        item['image_url'] = f"/media/dataset_telur_puyuh/Data_Testing_Otsu/{kelas}/{fname}"

    context = {
        'confusion_matrix': confusion_matrix_telur,
        'metrics_per_class': metrics_per_class,
        'overall_accuracy': overall_accuracy,
        'detail_results': data_hasil,
        'labels': labels  # Penting agar HTML bisa looping
    }

    return render(request, 'testing.html', context)


# ========================
# View klasifikasi gambar upload
# ========================
def klasifikasi_view(request):
    context = {}
    
    if request.method == 'POST' and request.FILES.get('gambar'):
        uploaded_file = request.FILES['gambar']

        try:
            # Buat folder penyimpanan
            klasifikasi_dir = os.path.join(settings.MEDIA_ROOT, 'gambar_klasifikasi')
            os.makedirs(klasifikasi_dir, exist_ok=True)

            # Simpan gambar upload
            fname = f"{uuid.uuid4().hex}_{uploaded_file.name}"
            fpath = os.path.join(klasifikasi_dir, fname)
            with open(fpath, 'wb+') as dest:
                for chunk in uploaded_file.chunks():
                    dest.write(chunk)

            # Proses grayscale dan otsu
            img = cv2.imread(fpath)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            gray_name = f"gray_{fname}"
            otsu_name = f"otsu_{fname}"
            gray_path = os.path.join(klasifikasi_dir, gray_name)
            otsu_path = os.path.join(klasifikasi_dir, otsu_name)
            cv2.imwrite(gray_path, gray)
            cv2.imwrite(otsu_path, otsu)

            # Ekstraksi fitur & klasifikasi
            fitur = hitung_fitur(fpath)
            hasil_klasifikasi = predict_image(fpath)  # ✅ cukup kirim path

            context = {
                'hasil': hasil_klasifikasi,
                'fitur': fitur,
                'k': 7,
                'gambar_url': settings.MEDIA_URL + f'gambar_klasifikasi/{fname}',
                'gray_url': settings.MEDIA_URL + f'gambar_klasifikasi/{gray_name}',
                'otsu_url': settings.MEDIA_URL + f'gambar_klasifikasi/{otsu_name}',
            }

        except Exception as e:
            context = {'error': f"Terjadi kesalahan saat klasifikasi: {str(e)}"}

    return render(request, 'klasifikasi.html', context)

def hitung_fitur(gambar_path):
    img = cv2.imread(gambar_path)
    img = cv2.resize(img, (338, 338))
    
    # Pisahkan channel RGB
    r_channel = img[:, :, 2]
    g_channel = img[:, :, 1]
    b_channel = img[:, :, 0]

    r_mean = np.mean(r_channel)
    g_mean = np.mean(g_channel)
    b_mean = np.mean(b_channel)

    # Ubah ke grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold Otsu
    otsu_threshold, otsu_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Hitung jumlah piksel hitam
    black_pixels = np.sum(otsu_result == 0)

    # Hitung fusi = jumlah kontur
    contours, _ = cv2.findContours(255 - otsu_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fusi_value = len(contours)

    fitur = {
        'R': round(r_mean, 2),
        'G': round(g_mean, 2),
        'B': round(b_mean, 2),
        'otsu': round(otsu_threshold, 2),
        'black': int(black_pixels),
        'fusi': fusi_value,
    }

    return fitur



def ekstrak_fitur_dataset(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (338, 338))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Rata-rata RGB
    r_mean = np.mean(img_resized[:, :, 2])
    g_mean = np.mean(img_resized[:, :, 1])
    b_mean = np.mean(img_resized[:, :, 0])

    # Threshold Otsu
    otsu_val, otsu_result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Jumlah piksel hitam
    black_pixel_count = np.sum(otsu_result == 0)

    # Fusi (jumlah kontur)
    contours, _ = cv2.findContours(255 - otsu_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fusi_value = len(contours)

    return {
        'ratar': round(r_mean, 4),
        'ratag': round(g_mean, 4),
        'ratab': round(b_mean, 4),
        'black': int(black_pixel_count),
        'fusi': fusi_value,
    }
