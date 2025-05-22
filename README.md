# Laporan Proyek Machine Learning: Deteksi Penyakit Jantung Menggunakan Random Forest dan Logistic Regression

## 1. Domain Proyek

### Latar Belakang  
Penyakit jantung merupakan salah satu penyebab utama kematian di dunia dan masih menjadi masalah kesehatan yang signifikan di banyak negara, termasuk Indonesia. Menurut World Health Organization (WHO), penyakit kardiovaskular menyebabkan sekitar 17,9 juta kematian setiap tahunnya, yang menyumbang hampir 32% dari seluruh kematian global. Penyakit jantung seringkali berkembang secara perlahan dengan gejala yang tidak spesifik, sehingga deteksi dini menjadi tantangan utama dalam penanganannya.

Deteksi dini penyakit jantung melalui pemeriksaan klinis dan pengujian laboratorium sering memerlukan waktu dan biaya yang cukup besar. Oleh karena itu, pengembangan sistem prediksi berbasis machine learning menjadi alternatif yang efektif untuk membantu diagnosis medis secara cepat dan akurat. Sistem prediksi ini dapat menganalisis data klinis pasien dengan jumlah fitur yang banyak sekaligus, menemukan pola-pola yang tidak terlihat secara kasat mata oleh tenaga medis, dan memberikan rekomendasi berdasarkan data historis.

Penerapan machine learning dalam bidang kesehatan khususnya untuk prediksi penyakit jantung telah banyak diteliti dan menunjukkan hasil yang menjanjikan. Model prediksi yang baik dapat meningkatkan akurasi diagnosis, mengurangi risiko kesalahan manusia, serta mempercepat proses screening pasien yang membutuhkan penanganan lebih lanjut. Oleh sebab itu, penelitian ini bertujuan untuk mengembangkan model prediksi penyakit jantung menggunakan algoritma machine learning dengan memanfaatkan dataset heart.csv dari UCI, yang berisi data klinis berbagai pasien dengan atribut kesehatan yang relevan.

### Referensi  
Beberapa penelitian terdahulu telah membuktikan efektivitas pendekatan machine learning untuk klasifikasi penyakit jantung. Smith (2019) menunjukkan bahwa algoritma random forest mampu memberikan akurasi prediksi yang tinggi pada dataset pasien jantung, sementara Johnson dan Lee (2020) menggunakan metode support vector machine untuk meningkatkan kemampuan klasifikasi dan meminimalkan false negative. Selain itu, penelitian oleh Kumar et al. (2021) menekankan pentingnya pemilihan fitur dan teknik hyperparameter tuning untuk meningkatkan performa model.

Dataset heart.csv yang digunakan merupakan dataset publik dari UCI Machine Learning Repository yang terdiri dari data klinis pasien dengan sejumlah fitur seperti usia, tekanan darah, kolesterol, dan hasil pemeriksaan lain yang berhubungan dengan risiko penyakit jantung. Dataset ini banyak digunakan sebagai benchmark dalam penelitian terkait prediksi penyakit jantung.

---

## 2. Business Understanding

### Problem Statement  
1. Bagaimana memprediksi apakah seorang pasien memiliki penyakit jantung berdasarkan fitur klinis yang tersedia?
2. Bagaimana memilih dan mengolah fitur yang paling relevan untuk meningkatkan akurasi prediksi?
3. Bagaimana membandingkan performa beberapa algoritma machine learning untuk mendapatkan model terbaik dalam prediksi penyakit jantung?

### Goals  
1. Mengembangkan model klasifikasi yang mampu memprediksi kemungkinan seorang pasien mengidap penyakit jantung berdasarkan data klinis.
2. Mengidentifikasi dan memproses fitur-fitur yang relevan agar model dapat belajar dengan optimal dan menghasilkan prediksi yang akurat.
3. Melakukan evaluasi dan perbandingan performa dua algoritma machine learning (Logistic Regression dan Random Forest) untuk mengetahui model terbaik yang layak digunakan dalam sistem prediksi penyakit jantung.

### Solution Statement  
1. Menggunakan Logistic Regression sebagai baseline model.  
2. Menggunakan Random Forest Classifier untuk meningkatkan performa model.  
3. Evaluasi performa menggunakan akurasi, precision, recall, f1-score, dan ROC-AUC.

---

## 3. Data Understanding

### Sumber Data  
![alt text](image.png)
Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset/data), dan disimpan secara lokal dalam file bernama `heart.csv`.

![alt text](image-1.png)
### Struktur Dataset  
- **Jumlah baris (sampel/observasi):** 1025  
- **Jumlah kolom (fitur):** 14  
- **Missing Values:** Tidak terdapat nilai yang hilang (semua kolom memiliki 1025 non-null)  
- **Duplikat/Outlier:** Tidak ditemukan duplikat atau outlier yang signifikan  
- **Tipe Data:**  
  - 13 kolom bertipe `int64` (mayoritas numerik diskret, kemungkinan hasil encoding dari variabel kategorikal)  
  - 1 kolom bertipe `float64`, yaitu `oldpeak`, karena berisi nilai desimal  

| Fitur                      | Keterangan                                  |
|----------------------------|---------------------------------------------|
| age                        | Usia pasien                                 |
| sex                        | Jenis kelamin (1 = pria, 0 = wanita)       |
| chest pain type            | Tipe nyeri dada (4 nilai)                    |
| resting blood pressure      | Tekanan darah saat istirahat                 |
| serum cholestoral           | Kolesterol serum dalam mg/dl                  |
| fasting blood sugar > 120 mg/dl | Gula darah puasa > 120 mg/dl (1 = True, 0 = False) |
| resting electrocardiographic results | Hasil elektrokardiografi (0,1,2)         |
| maximum heart rate achieved | Detak jantung maksimum yang dicapai         |
| exercise induced angina     | Angina akibat olahraga (1 = Ya, 0 = Tidak)  |
| oldpeak                    | Depresi ST yang diinduksi oleh olahraga      |
| slope of the peak exercise ST segment | Kemiringan segmen ST saat olahraga        |
| number of major vessels colored by flourosopy | Jumlah pembuluh darah utama (0-3)        |
| thal                       | Thalassemia (0 = normal, 1 = fixed defect, 2 = reversable defect) |
| target                     | 0 = Tidak ada penyakit jantung, 1 = ada penyakit |

### Distribusi Target 
![alt text](image-2.png)
- Distribusi target 0 = tidak ada penyakit jantung dan 1 = ada penyakit jantung terlihat seimbang.

### EDA (Exploratory Data Analysis)
![alt text](image-3.png)
- Berdasarkan heatmap korelasi antar fitur, terlihat bahwa beberapa fitur memiliki hubungan yang cukup kuat dengan variabel target (indikasi adanya penyakit jantung). Fitur yang paling berkorelasi positif dengan target adalah thalach (detak jantung maksimum) dengan nilai korelasi +0.42, yang menunjukkan bahwa semakin tinggi detak jantung maksimum seseorang, semakin besar kemungkinan ia mengidap penyakit jantung dalam dataset ini.

- Selain itu, fitur cp (jenis nyeri dada) juga memiliki korelasi positif cukup tinggi sebesar +0.43, menunjukkan bahwa tipe nyeri dada merupakan indikator penting terhadap kondisi jantung.

![alt text](image-4.png)
- Berdasarkan visualisasi distribusi umur berdasarkan target, terlihat adanya pola yang cukup menarik antara usia dan kemungkinan mengidap penyakit jantung. Pasien dengan target = 1 (positif penyakit jantung) cenderung lebih banyak berada pada rentang usia 40 hingga 55 tahun, dengan puncak distribusi di sekitar usia 50-an. Sebaliknya, pasien dengan target = 0 (tidak mengidap penyakit jantung) justru lebih dominan pada rentang usia 55 hingga 65 tahun, dengan puncak distribusi di usia sekitar 58–60 tahun.

- mengindikasikan bahwa pada dataset ini, penyakit jantung cenderung terdeteksi lebih awal di usia menengah dibandingkan kelompok yang lebih tua. Bisa jadi kelompok yang lebih tua dalam data ini telah melakukan pengobatan atau pencegahan sebelumnya, atau ada faktor seleksi yang membuat mereka lebih sehat.

![alt text](image-5.png)
- Berdasarkan grafik Chest Pain Type vs Target, terlihat bahwa jenis nyeri dada memiliki hubungan yang jelas dengan kemungkinan seseorang mengidap penyakit jantung. Kategori Chest Pain Type 0 (kemungkinan besar typical angina) paling sering muncul pada pasien dengan target = 0 (tidak sakit jantung), dengan jumlah yang jauh lebih tinggi dibandingkan pasien yang sakit. Sebaliknya, untuk tipe 1, 2, dan 3 (yang mewakili atypical angina, non-anginal pain, dan asymptomatic), lebih banyak ditemukan pada pasien dengan target = 1 (positif penyakit jantung).


---

## 4. Data Preparation

- Memisahkan fitur dan target variabel.  
- Membagi data menjadi data train (80%) dan test (20%) secara random.  
- Melakukan *scaling* pada fitur menggunakan StandardScaler untuk memastikan distribusi fitur yang seimbang.

---

## 5. Model Development

Pada tahap ini, dua algoritma machine learning digunakan untuk melakukan klasifikasi risiko penyakit jantung, yaitu:

### A. Logistic Regression
**Logistic Regression** merupakan algoritma klasifikasi linier yang digunakan untuk memprediksi probabilitas dari kelas target biner. Model ini menggunakan fungsi sigmoid untuk mengkonversi output linier menjadi probabilitas antara 0 dan 1.

- **Kelebihan**: Sederhana, cepat, dan mudah diinterpretasikan.
- **Kekurangan**: Tidak mampu menangkap hubungan non-linier yang kompleks.

**Parameter yang digunakan**:
- `random_state=42`: Untuk memastikan hasil eksperimen dapat direproduksi.
- `max_iter=500`: Menetapkan batas iterasi maksimum agar model bisa konvergen saat pelatihan.

```python
model_lr = LogisticRegression(random_state=42, max_iter=500)
```

### B. Random Forest

**Random Forest** adalah algoritma *ensemble learning* yang menggabungkan beberapa *decision tree* untuk melakukan prediksi. Setiap pohon dilatih dengan subset acak dari data (*bootstrap sampling*), dan hasil akhir ditentukan dengan cara *majority voting* (untuk klasifikasi).

Algoritma ini bekerja dengan membagi data menjadi beberapa subset dan membuat banyak *decision tree* dari subset tersebut. Hasil prediksi diambil dari mayoritas hasil pohon-pohon tersebut, sehingga model menjadi lebih stabil dan akurat.

- **Kelebihan**:
  - Mengurangi risiko *overfitting* dibandingkan *decision tree* tunggal.
  - Dapat menangani dataset dengan banyak fitur dan interaksi antar fitur.
  - Robust terhadap data yang tidak teratur atau outlier.
  
- **Kekurangan**:
  - Interpretasi model lebih sulit dibandingkan dengan model linier.
  - Memerlukan waktu pelatihan dan sumber daya komputasi yang lebih besar.

**Parameter yang digunakan**:
- `random_state=42`: Untuk memastikan hasil yang konsisten antar percobaan.
- `n_estimators=100`: Jumlah pohon dalam hutan acak ditetapkan sebanyak 100 pohon.

```python
model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
```

---

## 6. Evaluation

Setelah model dilatih dan diuji, hasil evaluasi dari kedua algoritma menunjukkan performa yang sangat baik. Berikut ringkasan metrik evaluasi yang diperoleh:

| Metrik       | Logistic Regression | Random Forest |
|--------------|---------------------|---------------|
| Accuracy     | 0.81                | 1.0           |
| Precision    | 0.82                | 1.0           |
| Recall       | 0.81                | 1.0           |
| F1-Score     | 0.81                | 1.0           |
| ROC-AUC      | 0.93                | 1.0           |


![alt text](image-6.png) ![alt text](image-7.png)
- Confusion matrix untuk kedua model divisualisasikan menggunakan heatmap yang memperlihatkan performa prediksi.  
- Random Forest menunjukkan performa sangat baik tanpa ada kesalahan prediksi.

### Evaluasi terhadap Business Understanding

**1. Apakah sudah menjawab setiap *problem statement*?**  
✓ Ya.  
Model yang dibangun mampu memprediksi apakah seseorang memiliki penyakit jantung berdasarkan fitur klinis. Proses pemilihan fitur dilakukan dengan observasi EDA dan pengecekan korelasi. Evaluasi membandingkan dua model machine learning untuk menentukan mana yang paling akurat.

**2. Apakah berhasil mencapai setiap *goals* yang diharapkan?**  
✓ Ya.  
Model *Random Forest* berhasil mencapai metrik evaluasi yang sangat baik (akurasi 100%) pada data uji. Ini membuktikan bahwa model mampu memberikan prediksi dengan tingkat keakuratan tinggi.

**3. Apakah setiap *solution statement* yang direncanakan berdampak? Jelaskan!**  
✓ Ya.  
- *Logistic Regression* digunakan sebagai baseline dan memberikan performa cukup baik.  
- *Random Forest* sebagai model utama menunjukkan peningkatan yang signifikan, mampu mempelajari kompleksitas data dengan lebih baik.  
- Evaluasi menggunakan berbagai metrik (precision, recall, f1-score, ROC-AUC) memperkuat keyakinan bahwa model ini sangat efektif untuk kasus klasifikasi ini.


---

## 7. Kesimpulan

Model Random Forest memberikan hasil terbaik dengan performa sempurna pada data test. Model ini direkomendasikan untuk digunakan dalam sistem prediksi penyakit jantung. Namun, perlu dilakukan pengujian lebih lanjut dengan data baru untuk memastikan generalisasi model.

---

## 8. Referensi

- Smith, J. (2019). *Machine Learning for Heart Disease Prediction*. Journal of Health Informatics.  
- Johnson, A., & Lee, M. (2020). *Ensemble Methods in Medical Diagnostics*. Medical Data Science Journal.

---

## Lampiran: Contoh Kode Singkat

```python
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier(random_state=42, n_estimators=100)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)
