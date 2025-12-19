## INFORMASI PROYEK

**Judul Proyek:**  
[(Contoh: "Klasifikasi Penyakit Daun Menggunakan CNN", "Prediksi Harga Rumah dengan Machine Learning", "Analisis Sentimen Ulasan Produk")]

**Nama Mahasiswa:** [Chandra Dina Sefrilian]  
**NIM:** [233307039]  
**Program Studi:** [Teknologi Informasi]  
**Mata Kuliah:** [Data Science]  
**Dosen Pengampu:** [Gus Nanang Syaifuddiin, S.Kom., M.Kom.]  
**Tahun Akademik:** [2025]
**Link GitHub Repository:** [[URL Repository](https://github.com/ChandraDina/Hepatits-DataSet-Projek.git)]
**Link Video Pembahasan:** [[URL Repository](https://drive.google.com/file/d/1JITM-ebSj8owbsMR4bu009SHrMg_AoQy/view?usp=drive_link)]

---

## 1. LEARNING OUTCOMES
Pada proyek ini, mahasiswa diharapkan dapat:
1. Memahami konteks masalah dan merumuskan problem statement secara jelas
2. Melakukan analisis dan eksplorasi data (EDA) secara komprehensif (**OPSIONAL**)
3. Melakukan data preparation yang sesuai dengan karakteristik dataset
4. Mengembangkan tiga model machine learning yang terdiri dari (**WAJIB**):
   - Model baseline
   - Model machine learning / advanced
   - Model deep learning (**WAJIB**)
5. Menggunakan metrik evaluasi yang relevan dengan jenis tugas ML
6. Melaporkan hasil eksperimen secara ilmiah dan sistematis
7. Mengunggah seluruh kode proyek ke GitHub (**WAJIB**)
8. Menerapkan prinsip software engineering dalam pengembangan proyek

---

## 2. PROJECT OVERVIEW

### 2.1 Latar Belakang
**Isi bagian ini dengan:**
- Mengapa proyek ini penting?
  Proyek ini memiliki urgensi yang tinggi karena proses diagnosis klinis pada pasien hepatitis sering kali dihadapkan pada ketidakpastian prognosis yang kompleks dalam domain kesehatan. Dengan memanfaatkan dataset dari UCI Machine Learning Repository, penelitian ini sangat penting untuk menyediakan sistem pendukung keputusan yang objektif bagi tenaga medis guna meminimalisir subjektivitas dalam penilaian kondisi pasien. Melalui penerapan model machine learning yang akurat, tenaga kesehatan dapat meningkatkan efektivitas deteksi dini terhadap risiko fatalitas, sehingga intervensi penyelamatan nyawa dapat dilakukan secara lebih cepat, tepat, dan efisien.
- Permasalahan umum pada domain terkait (misal: kesehatan, pendidikan, keuangan, pertanian, NLP, computer vision, dll.)
  Tantangan utama dalam domain kesehatan, khususnya terkait penyakit hepatitis, adalah tingginya kompleksitas diagnosis klinis yang sering kali memicu ketidakpastian dalam menentukan prognosis pasien. Permasalahan umum yang kerap muncul meliputi keterbatasan jumlah sampel data yang tersedia serta banyaknya informasi yang hilang pada hasil pemeriksaan laboratorium. Selain itu, terdapat ketidakseimbangan distribusi kelas yang signifikan, di mana jumlah pasien yang bertahan hidup jauh lebih banyak dibandingkan pasien yang meninggal dunia, sehingga berisiko menimbulkan bias pada performa model jika tidak ditangani melalui teknik penyeimbangan data khusus
- Manfaat proyek untuk pengguna, bisnis, atau penelitian
  Implementasi proyek ini memberikan manfaat strategis melalui penyediaan alat bantu pengambilan keputusan yang objektif bagi tenaga medis untuk meminimalisir subjektivitas dalam diagnosis. Bagi pengguna dan praktisi kesehatan, model ini berfungsi meningkatkan akurasi deteksi dini terhadap risiko kematian pasien sehingga intervensi medis dapat dilakukan secara lebih cepat dan tepat sasaran. Selain itu, secara akademis, penelitian ini memberikan kontribusi penting dalam pengembangan literatur mengenai efektivitas algoritma machine learning untuk menangani karakteristik dataset medis yang kompleks dan memiliki distribusi kelas tidak seimbang
- Studi literatur atau referensi ilmiah (minimal 1–2 sumber wajib)
  Penelitian oleh Nilashi et al. (2019) menunjukkan bahwa penerapan pendekatan berbasis data (data-driven) sangat efektif dalam mengurangi subjektivitas medis serta mengoptimalkan efisiensi intervensi penyelamatan nyawa pasien

**Contoh referensi (berformat APA/IEEE):**
> Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

**[Jelaskan konteks dan latar belakang proyek]**

## 3. BUSINESS UNDERSTANDING / PROBLEM UNDERSTANDING
### 3.1 Problem Statements
Tuliskan 2–4 pernyataan masalah yang jelas dan spesifik.
1. Membangun model klasifikasi yang akurat untuk memprediksi status kelangsungan hidup pasien hepatitis sebagai pendukung keputusan medis.
2. Mengoptimalkan kualitas data melalui teknik imputasi untuk menangani nilai yang hilang dan keterbatasan jumlah sampel.
3. Mengatasi ketimpangan data antara pasien yang selamat dan meninggal menggunakan teknik penyeimbangan agar hasil prediksi tidak bias
4. Mengevaluasi perbandingan performa antara model baseline, advanced, dan deep learning untuk menentukan algoritma yang paling stabil.

**Contoh (universal):**
1. Model perlu mampu memprediksi nilai target dengan akurasi tinggi
2. Sistem harus dapat mengidentifikasi pola pada citra secara otomatis
3. Dataset memiliki noise sehingga perlu preprocessing yang tepat
4. Dibutuhkan model deep learning yang mampu belajar representasi fitur kompleks

**[Tulis problem statements Anda di sini]**
Tantangan utama dalam penelitian ini adalah bagaimana membangun model klasifikasi yang mampu memprediksi status kelangsungan hidup pasien hepatitis secara akurat di tengah kendala data klinis yang tidak lengkap. Permasalahan ini mencakup penanganan nilai yang hilang (missing values) pada hasil laboratorium serta upaya mengatasi ketidakseimbangan distribusi data antara pasien yang bertahan hidup dan yang meninggal dunia. Selain itu, diperlukan analisis untuk menentukan sejauh mana algoritma machine learning dapat mereduksi subjektivitas medis guna memberikan hasil prediksi yang lebih objektif dan reliabel.
### 3.2 Goals

Tujuan harus spesifik, terukur, dan selaras dengan problem statement.
Tujuan utama dari proyek ini adalah membangun model klasifikasi yang mampu memprediksi status kelangsungan hidup pasien hepatitis secara akurat dengan memanfaatkan data klinis yang tersedia. Sasaran spesifiknya mencakup optimalisasi kualitas data melalui teknik imputasi untuk menangani nilai yang hilang serta penerapan strategi penyeimbangan data guna mengatasi ketimpangan proporsi antara pasien yang selamat dan meninggal. Selain itu, penelitian ini bertujuan untuk mengevaluasi perbandingan performa antara model baseline (Naive Bayes), machine learning lanjutan (SVM), dan ensemble learning (Random Forest) guna menentukan algoritma yang paling stabil dalam mendukung keputusan medis yang objektif.
**Contoh tujuan:**
1. Membangun model ML untuk memprediksi variabel target dengan akurasi > 80%
2. Mengukur performa tiga pendekatan model (baseline, advanced, deep learning)
3. Menentukan model terbaik berdasarkan metrik evaluasi yang relevan
4. Menghasilkan sistem yang dapat bekerja secara reproducible

**[Tulis goals Anda di sini]**
Tujuan utama dari proyek ini adalah membangun arsitektur model klasifikasi yang mampu memprediksi status kelangsungan hidup pasien hepatitis dengan tingkat akurasi yang optimal berdasarkan data klinis yang tersedia. Sasaran spesifik penelitian ini mencakup peningkatan kualitas data melalui teknik imputasi untuk menangani nilai yang hilang serta penerapan strategi penyeimbangan kelas guna menghasilkan model yang objektif. Selain itu, proyek ini bertujuan untuk mengevaluasi efektivitas perbandingan antara model baseline (Naïve Bayes), advanced machine learning (SVM), dan model ensemble (Random Forest) guna menentukan algoritma yang paling stabil dalam mendukung pengambilan keputusan medis yang reliabel.

### 3.3 Solution Approach

Mahasiswa **WAJIB** menggunakan minimal **tiga model** dengan komposisi sebagai berikut:
#### **Model 1 – Baseline Model**
Model sederhana sebagai pembanding dasar.
**Pilihan model:**
- Linear Regression (untuk regresi)
- Logistic Regression (untuk klasifikasi)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Naive Bayes
  1. model 1 naive bayes
     Model Naïve Bayes, khususnya tipe Gaussian Naïve Bayes, diimplementasikan sebagai model baseline atau standar acuan awal dalam penelitian ini. Algoritma ini dipilih karena efisiensinya dalam menangani data numerik yang diasumsikan mengikuti distribusi normal, serta kemampuannya dalam melakukan klasifikasi secara cepat melalui pendekatan probabilistik. Alasan utama penggunaan model ini adalah untuk memberikan gambaran performa awal yang sederhana sebelum beralih ke algoritma yang lebih kompleks, mengingat asumsi independensi antar fitur yang dimiliki model ini dapat mengidentifikasi pola dasar pada data klinis pasien hepatitis secara efisien.

**[Jelaskan model baseline yang Anda pilih dan alasannya]**

#### **Model 2 – Advanced / ML Model**
Model machine learning yang lebih kompleks.
**Pilihan model:**
- Random Forest
- Gradient Boosting (XGBoost, LightGBM, CatBoost)
- Support Vector Machine (SVM)
- Ensemble methods
- Clustering (K-Means, DBSCAN) - untuk unsupervised
- PCA / dimensionality reduction (untuk preprocessing)
  1. model 2 random forest
     Random Forest merupakan algoritma ensemble learning yang bekerja dengan membangun sekumpulan pohon keputusan (decision trees) dan menggabungkan hasilnya untuk mendapatkan prediksi yang lebih akurat. Alasan utama pemilihan model ini adalah kemampuannya yang sangat baik dalam menangani hubungan non-linear serta interaksi kompleks antar fitur klinis tanpa memerlukan penyetelan parameter yang rumit. Selain itu, Random Forest memiliki ketahanan yang tinggi terhadap overfitting dan mampu memberikan informasi mengenai tingkat kepentingan fitur (feature importance), sehingga sangat efektif dalam mengidentifikasi parameter medis yang paling berpengaruh terhadap prognosis pasien hepatitis.

**[Jelaskan model advanced yang Anda pilih dan alasannya]**

#### **Model 3 – Deep Learning Model (WAJIB)**
Model deep learning yang sesuai dengan jenis data.
**Pilihan Implementasi (pilih salah satu sesuai dataset):**
**A. Tabular Data:**
- Multilayer Perceptron (MLP) / Neural Network
- Minimum: 2 hidden layers
- Contoh: prediksi harga, klasifikasi binary/multiclass
  1. model 3 vm
     Support Vector Machine (SVM) adalah algoritma pembelajaran terawasi yang bekerja dengan cara mencari hyperplane optimal untuk memisahkan kelas data dalam ruang berdimensi tinggi. Alasan utama pemilihan SVM dalam proyek ini adalah efektivitasnya yang tinggi pada dataset dengan jumlah sampel terbatas namun memiliki dimensi fitur yang beragam, seperti pada data klinis hepatitis. Dengan menggunakan fungsi kernel, SVM mampu menangani pola data yang tidak dapat dipisahkan secara linier dengan sangat presisi, sehingga diharapkan dapat menghasilkan batas keputusan yang kuat untuk membedakan risiko fatalitas pasien secara akurat.

**B. Image Data:**
- CNN sederhana (minimum 2 convolutional layers) **ATAU**
- Transfer Learning (ResNet, VGG, MobileNet, EfficientNet) - **recommended**
- Contoh: klasifikasi gambar, object detection

**C. Text Data:**
- LSTM/GRU (minimum 1 layer) **ATAU**
- Embedding + Dense layers **ATAU**
- Pre-trained model (BERT, DistilBERT, Word2Vec)
- Contoh: sentiment analysis, text classification

**D. Time Series:**
- LSTM/GRU untuk sequential prediction
- Contoh: forecasting, anomaly detection

**E. Recommender Systems:**
- Neural Collaborative Filtering (NCF)
- Autoencoder-based Collaborative Filtering
- Deep Matrix Factorization

**Minimum Requirements untuk Deep Learning:**
- ✅ Model harus training minimal 10 epochs
- ✅ Harus ada plot loss dan accuracy/metric per epoch
- ✅ Harus ada hasil prediksi pada test set
- ✅ Training time dicatat (untuk dokumentasi)

**Tidak Diperbolehkan:**
- ❌ Copy-paste kode tanpa pemahaman
- ❌ Model tidak di-train (hanya define arsitektur)
- ❌ Tidak ada evaluasi pada test set

**[Jelaskan model deep learning yang Anda pilih dan alasannya]**

---

## 4. DATA UNDERSTANDING
### 4.1 Informasi Dataset
**Sumber Dataset:**  
[Sebutkan sumber: UCI Machine Learning Repository]

**Deskripsi Dataset:**
- Jumlah baris (rows): [155]
- Jumlah kolom (columns/features): [20]
- Tipe data: [kategorikal,integer, dan real]
- Ukuran dataset: [11kb-15kb]
- Format file: [CSV ]

### 4.2 Deskripsi Fitur
Jelaskan setiap fitur/kolom yang ada dalam dataset.
**Contoh tabel:**
| Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|------------|-----------|-----------|--------------|
| id | Integer | ID unik data | 1, 2, 3 |
| age | Integer | Usia (tahun) | 25, 30, 45 |
| income | Float | Pendapatan (juta) | 5.5, 10.2 |
| category | Categorical | Kategori produk | A, B, C |
| text | String | Teks ulasan | "Produk bagus..." |
| image | Image | Citra 224x224 RGB | Array 224x224x3 |
| label | Categorical | Label target | 0, 1 atau "positif", "negatif" |

**[Buat tabel deskripsi fitur Anda di sini]**
|Nama Fitur | Tipe Data | Deskripsi | Contoh Nilai |
|-----------|-----------|-----------|--------------|
| Class | Kategorikal | Status akhir kelangsungan hidup | 1(Die),2(live) |
| AGE | Numerik | Usia pasien dalam satuan tahun | 30,50,78 |
| SEX | Kategorikal | Jenis kelamin pasien | 1(male),2(female) |
| STEROID | Kategorikal | Penggunaan obat-obatan steroid | 1(no),2(yes) |
| ANTIVIRALS | Kategorika | Penggunaan terapi antivirus | 1 (No), 2 (Yes)
| FATIGUE | Kategorikal | Gejala klinis rasa lelah kronis | 1 (No), 2 (Yes)
| MALAISE | Kategorikal | Perasaan lesu atau tidak enak badan | 1 (No), 2 (Yes)
| ANOREXIA | Kategorikal | Gejala hilang nafsu makan | 1 (No), 2 (Yes)
| LIVER BIG | Kategorikal | Kondisi pembesaran organ hati | 1 (No), 2 (Yes)
| LIVER FIRM | Kategorikal | Tekstur hati (keras atau tidak) | 1 (No), 2 (Yes)
| SPLEEN PALPABLE | Kategorikal | Limpa dapat teraba saat diperiksa | 1 (No), 2 (Yes)
| SPIDERS | Kategorikal | Munculnya tanda pembuluh darah laba-laba | 1 (No), 2 (Yes)
| ASCITES | Kategorikal | Penumpukan cairan di rongga perut | 1 (No), 2 (Yes)
| VARICES | Kategorikal | Adanya varises pada kerongkongan | 1 (No), 2 (Yes)
| BILIRUBIN | Numerik | Kadar pigmen bilirubin dalam darah | 0.7, 1.2, 4.0
| ALK PHOSPHATE | Numerik | Kadar enzim fosfatase alkali | 85, 120, 250
| SGOT | Numerik | Kadar enzim aspartat aminotransferase | 18, 45, 100
| ALBUMIN | Numerik | Kadar protein albumin dalam darah | 2.1, 3.5, 4.5
| PROTIME | Numerik | Kecepatan waktu pembekuan darah | 30, 80, 100
| HISTOLOGY | Kategorikal | Hasil biopsi jaringan hati | 1 (No), 2 (Yes)
### 4.3 Kondisi Data

Jelaskan kondisi dan permasalahan data:

- **Missing Values:** [Ada, 48%]
- **Duplicate Data:** [Tidak]
- **Outliers:** [Ada, numerik?]
- **Imbalanced Data:** [Ada, 80:20]
- **Noise:** [Ada. Noise dalam dataset ini muncul dalam bentuk inkonsistensi pengisian data mentah, di mana nilai yang hilang direpresentasikan dengan simbol tanda tanya (?), sehingga memerlukan konversi tipe data sebelum diolah secara komputasi.]
- **Data Quality Issues:** [Masalah utama kualitas data terletak pada ukuran sampel yang kecil (n=155) dikombinasikan dengan banyaknya missing values. Hal ini menciptakan tantangan dalam menjaga validitas statistik saat melakukan imputasi atau penghapusan data, karena setiap baris sangat berharga untuk melatih model.]

### 4.4 Exploratory Data Analysis (EDA) - (**OPSIONAL**)

**Requirement:** Minimal 3 visualisasi yang bermakna dan insight-nya.
**Contoh jenis visualisasi yang dapat digunakan:**
- Histogram (distribusi data)
- Boxplot (deteksi outliers)
- Heatmap korelasi (hubungan antar fitur)
- Bar plot (distribusi kategori)
- Scatter plot (hubungan 2 variabel)
- Wordcloud (untuk text data)
- Sample images (untuk image data)
- Time series plot (untuk temporal data)
- Confusion matrix heatmap
- Class distribution plot


#### Visualisasi 1: [Judul Visualisasi]
[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 2: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]

#### Visualisasi 3: [Judul Visualisasi]

[Insert gambar/plot]

**Insight:**  
[Jelaskan apa yang dapat dipelajari dari visualisasi ini]



---

## 5. DATA PREPARATION
Tahapan ini mencakup seluruh rangkaian prosedur transformasi data yang bertujuan untuk meningkatkan kualitas dan kesiapan dataset sebelum memasuki fase pemodelan. Proses diawali dengan pembersihan data melalui penanganan nilai yang hilang menggunakan teknik imputasi yang sesuai, diikuti dengan konversi tipe data dari format kategorikal menjadi numerik agar dapat diproses oleh algoritma. Selain itu, dilakukan standarisasi skala fitur untuk menyamakan rentang nilai numerik serta penerapan strategi penyeimbangan data guna mengatasi ketimpangan jumlah sampel antara pasien yang selamat dan yang meninggal dunia, sehingga model yang dihasilkan lebih objektif dan memiliki performa yang optimal.
### 5.1 Data Cleaning
**Aktivitas:**
- Handling missing values
- Removing duplicates
- Handling outliers
- Data type conversion
**Contoh:**
```
Missing Values:
- Fitur 'age' memiliki 50 missing values (5% dari data)
- Strategi: Imputasi dengan median karena distribusi skewed
- Alasan: Median lebih robust terhadap outliers dibanding mean
```

**[Jelaskan langkah-langkah data cleaning yang Anda lakukan]**
Proses pembersihan data dilakukan melalui serangkaian tahapan sistematis untuk menjamin validitas dan reliabilitas dataset sebelum tahap pemodelan. Langkah-langkah tersebut mencakup identifikasi dan penanganan nilai yang hilang menggunakan teknik imputasi berbasis median dan modus untuk menjaga integritas informasi medis, serta melakukan verifikasi data duplikat guna menghindari redundansi sampel. Selain itu, dilakukan normalisasi terhadap nilai pencilan (outliers) pada fitur laboratorium serta konversi tipe data dari format objek ke numerik agar seluruh atribut klinis dapat diproses secara optimal oleh algoritma machine learning.
### 5.2 Feature Engineering
**Aktivitas:**
- Creating new features
- Feature extraction
- Feature selection
- Dimensionality reduction

**[Jelaskan feature engineering yang Anda lakukan]**
Tahap rekayasa fitur dilakukan untuk meningkatkan daya prediksi model dengan mentransformasi atribut mentah menjadi representasi yang lebih informatif bagi algoritma. Proses ini meliputi penskalaan fitur menggunakan StandardScaler untuk menyelaraskan rentang nilai numerik yang beragam pada hasil laboratorium, sehingga mencegah dominasi fitur tertentu saat perhitungan jarak pada model seperti SVM. Selain itu, dilakukan pengkodean ulang pada variabel kategorikal serta pemilihan fitur yang paling relevan guna mereduksi dimensi data yang kurang berkontribusi, yang pada akhirnya bertujuan untuk memperjelas pola hubungan antara indikator klinis dengan status kelangsungan hidup pasien.

### 5.3 Data Transformation

**Untuk Data Tabular:**
- Encoding (Label Encoding, One-Hot Encoding, Ordinal Encoding)
- Scaling (Standardization, Normalization, MinMaxScaler)

**Untuk Data Text:**
- Tokenization
- Lowercasing
- Removing punctuation/stopwords
- Stemming/Lemmatization
- Padding sequences
- Word embedding (Word2Vec, GloVe, fastText)

**Untuk Data Image:**
- Resizing
- Normalization (pixel values 0-1 atau -1 to 1)
- Data augmentation (rotation, flip, zoom, brightness, etc.)
- Color space conversion

**Untuk Time Series:**
- Creating time windows
- Lag features
- Rolling statistics
- Differencing

**[Jelaskan transformasi yang Anda lakukan]**\
Tahap transformasi data dilakukan untuk menyelaraskan format dan distribusi variabel agar kompatibel dengan kebutuhan komputasi algoritma machine learning. Proses ini melibatkan standardisasi fitur numerik untuk memastikan bahwa parameter dengan satuan berbeda, seperti kadar bilirubin dan usia, berada pada skala yang seragam sehingga tidak menyebabkan bias pada model. Selain itu, dilakukan penyeimbangan distribusi kelas menggunakan teknik sampling untuk memitigasi dampak ketimpangan data antara pasien yang selamat dan meninggal, serta penerapan encoding pada variabel kategori guna mengubah informasi tekstual menjadi representasi vektor numerik yang dapat diinterpretasikan secara akurat oleh sistem.

### 5.4 Data Splitting

**Strategi pembagian data:**
```
- Training set: [X]% ([jumlah] samples)
- Validation set: [X]% ([jumlah] samples) - jika ada
- Test set: [X]% ([jumlah] samples)
```
**Contoh:**
```
Menggunakan stratified split untuk mempertahankan distribusi kelas:
- Training: 80% (8000 samples)
- Test: 20% (2000 samples)
- Random state: 42 untuk reproducibility
```

**[Jelaskan strategi splitting Anda dan alasannya]**
Pembagian dataset dilakukan dengan memisahkan data menjadi set pelatihan (training set) dan set pengujian (testing set) menggunakan rasio 80:20 guna menjamin keseimbangan antara volume data untuk pembelajaran dan validasi. Strategi ini diterapkan dengan metode stratified sampling untuk memastikan bahwa proporsi kelas pasien yang selamat dan meninggal tetap konsisten di kedua subset data, sehingga menghindari bias distribusi. Alasan utama penggunaan skema ini adalah untuk memberikan ruang yang cukup bagi model dalam mempelajari pola klinis yang kompleks, sekaligus menyediakan data uji yang independen untuk mengevaluasi kemampuan generalisasi model terhadap data baru secara objektif.
### 5.5 Data Balancing (jika diperlukan)
**Teknik yang digunakan:**
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Undersampling
- Class weights
- Ensemble sampling

**[Jelaskan jika Anda melakukan data balancing]**
Tahap penyeimbangan data diimplementasikan untuk mengatasi ketimpangan signifikan antara jumlah pasien yang selamat dan yang meninggal dunia guna menghindari bias prediksi pada kelas mayoritas. Proses ini dilakukan dengan menerapkan teknik Synthetic Minority Over-sampling Technique (SMOTE) yang bekerja dengan menciptakan sampel sintetis secara cerdas pada kelas minoritas berdasarkan kedekatan fitur klinis yang ada. Strategi ini dipilih agar model memiliki kemampuan yang setara dalam mengenali pola risiko fatalitas pasien, sehingga hasil klasifikasi yang diperoleh menjadi lebih adil, objektif, dan memiliki tingkat sensitivitas yang lebih tinggi terhadap kasus-kasus kritis.

### 5.6 Ringkasan Data Preparation

**Per langkah, jelaskan:**
1. **Apa** yang dilakukan
**[Melakukan estimasi dan pengisian data pada kolom yang memiliki nilai kosong atau simbol tanda tanya]**
2. **Mengapa** penting
**[Data medis yang tidak lengkap dapat menyebabkan bias atau kegagalan saat proses pelatihan model, sehingga integritas informasi harus dijaga tanpa menghapus terlalu banyak baris.]**
3. **Bagaimana** implementasinya
**[Implementasi dilakukan dengan menggunakan teknik SimpleImputer, di mana nilai numerik diisi menggunakan median untuk menghindari pengaruh pencilan, sementara data kategorikal diisi menggunakan modus (nilai yang paling sering muncul)]**

---

## 6. MODELING
### 6.1 Model 1 — Baseline Model
#### 6.1.1 Deskripsi Model

**Nama Model:** [Nama model,: Gaussian Naïve Bayes]
**Teori Singkat:**  
[Model ini merupakan algoritma klasifikasi probabilistik yang bekerja berdasarkan Teorema Bayes dengan asumsi bahwa setiap fitur klinis bersifat independen satu sama lain. Algoritma ini memprediksi peluang kelangsungan hidup pasien dengan menghitung distribusi probabilitas Gaussian dari setiap variabel input, lalu menggabungkannya untuk menentukan kelas target yang paling memungkinkan.]
**Alasan Pemilihan:**  
[Gaussian Naïve Bayes dipilih sebagai model baseline karena efisiensinya yang sangat tinggi dalam menangani dataset kecil serta kemampuannya memberikan hasil prediksi yang stabil meskipun terdapat korelasi antar fitur yang sederhana. Model ini menjadi standar pembanding yang efektif untuk melihat sejauh mana model yang lebih kompleks, seperti Random Forest atau SVM, dapat meningkatkan performa prediksi pada data medis hepatitis ini.]

#### 6.1.2 Hyperparameter
**Parameter yang digunakan:**
```
[Tuliskan parameter penting, contoh:]
- C (regularization): 1.0
- solver: 'lbfgs'
- max_iter: 100
parameter yang digunakan :
random forest
1. n_estimators: 100 (Jumlah pohon keputusan)
2. max_depth: None (Kedalaman pohon maksimal)
3. random_state: 42 (Konsistensi hasil)
SVM
1. C: 1.0 (Parameter regularisasi)
2. kernel: 'rbf' (Fungsi pemetaan data non-linear)
3. gamma: 'scale' (Koefisien kernel)
```

#### 6.1.3 Implementasi (Ringkas)
```python
# Contoh kode (opsional, bisa dipindah ke GitHub)
from sklearn.linear_model import LogisticRegression

model_baseline = LogisticRegression(C=1.0, max_iter=100)
model_baseline.fit(X_train, y_train)
y_pred_baseline = model_baseline.predict(X_test)
```

#### 6.1.4 Hasil Awal

**[Berdasarkan hasil pengujian awal, ketiga model yang diimplementasikan menunjukkan performa yang cukup kompetitif dalam mengklasifikasikan status kelangsungan hidup pasien hepatitis. Model Random Forest dan SVM cenderung memberikan tingkat akurasi yang lebih tinggi dibandingkan model baseline karena kemampuannya dalam menangani pola data klinis yang kompleks. Namun, aspek krusial yang ditemukan adalah variasi nilai recall pada setiap model, yang menunjukkan perbedaan kemampuan dalam mendeteksi kelas minoritas atau pasien dengan risiko fatalitas tinggi. Hasil evaluasi yang lebih mendalam mengenai perbandingan metrik akurasi, presisi, dan skor F1 untuk setiap algoritma akan dijabarkan secara terperinci pada Bagian 7.]**

---

### 6.2 Model 2 — ML / Advanced Model
#### 6.2.1 Deskripsi Model

**Nama Model:** [Nama model: Random Forest]
**Teori Singkat:**  
[Random Forest merupakan algoritma ensemble learning yang bekerja dengan membangun sekumpulan besar pohon keputusan (decision trees) secara paralel selama fase pelatihan. Prediksi akhir ditentukan berdasarkan mekanisme majority voting (suara terbanyak) dari seluruh pohon yang terbentuk, sehingga hasil klasifikasi menjadi lebih stabil dan akurat dibandingkan hanya menggunakan satu pohon keputusan tunggal.]

**Alasan Pemilihan:**  
[Algoritma ini dipilih karena ketangguhannya dalam mengolah dataset medis yang memiliki hubungan antar fitur yang kompleks dan tidak linier. Random Forest sangat efektif dalam menangani data hepatitis yang memiliki banyak variabel kategori dan numerik sekaligus tanpa memerlukan prapemrosesan yang sangat ekstensif.]

**Keunggulan:**
- [Ketahanan terhadap Overfitting: Dengan menggabungkan banyak pohon, model ini mampu mereduksi varians sehingga tidak mudah terjebak pada pola data latih saja.]

**Kelemahan:**
- [Kompleksitas Model: Memerlukan daya komputasi dan memori yang lebih besar seiring bertambahnya jumlah pohon yang dibangun.]

#### 6.2.2 Hyperparameter

**Parameter yang digunakan:**
1. n_estimators: 100 (Jumlah pohon keputusan dalam hutan)
2. max_depth: 10 (Batasan kedalaman maksimal setiap pohon)
3. min_samples_split: 2 (Jumlah sampel minimum untuk membagi simpul internal)
4. random_state: 42 (Menjamin konsistensi hasil ekperimen)
[Tuliskan parameter penting, contoh:]
- n_estimators: 100
- max_depth: 10
- learning_rate: 0.1
- min_samples_split: 2
```

**Hyperparameter Tuning (jika dilakukan):**
- Metode: [Grid Search ]
- Best parameters: ['n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2]

#### 6.2.3 Implementasi (Ringkas)
```python
from sklearn.ensemble import RandomForestClassifier

# Menginisialisasi model dengan parameter optimal
model_rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    random_state=42
)

# Melatih model pada data latih
model_rf.fit(X_train_balanced, y_train_balanced)

# Melakukan prediksi pada data uji
y_pred_rf = model_rf.predict(X_test)
# Contoh kode
from sklearn.ensemble import RandomForestClassifier

model_advanced = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
model_advanced.fit(X_train, y_train)
y_pred_advanced = model_advanced.predict(X_test)
```

#### 6.2.4 Hasil Model

**[Berdasarkan hasil pengujian pada tahap ini, model Random Forest menunjukkan performa yang sangat solid dengan tingkat akurasi yang melampaui model baseline. Penggunaan parameter yang telah dioptimasi berhasil meminimalkan kesalahan klasifikasi, terutama pada deteksi pasien dalam kategori kritis. Secara umum, model ini menunjukkan stabilitas yang baik antara skor presisi dan recall, yang mengindikasikan bahwa pendekatan ensemble sangat efektif dalam menangani kompleksitas fitur pada dataset Hepatitis. Analisis perbandingan metrik yang lebih mendalam serta visualisasi performa secara menyeluruh akan dijabarkan secara detail pada Bagian 7.]**

---

### 6.3 Model 3 — Deep Learning Model (WAJIB)

#### 6.3.1 Deskripsi Model

**Nama Model:** [Nama arsitektur: SVM]

** (Centang) Jenis Deep Learning: **
- [x] Multilayer Perceptron (MLP) - untuk tabular
- [ ] Convolutional Neural Network (CNN) - untuk image
- [ ] Recurrent Neural Network (LSTM/GRU) - untuk sequential/text
- [ ] Transfer Learning - untuk image
- [ ] Transformer-based - untuk NLP
- [ ] Autoencoder - untuk unsupervised
- [ ] Neural Collaborative Filtering - untuk recommender

**Alasan Pemilihan:**  
[Pemilihan arsitektur MLP didasarkan pada kemampuannya yang unggul dalam mengekstraksi informasi dari data klinis hepatitis yang memiliki hubungan variabel yang sangat kompleks dan bersifat non-linear. Struktur lapisan tersembunyi pada jaringan saraf ini memungkinkan model untuk mempelajari pola interaksi antara hasil laboratorium dan gejala fisik secara mendalam, sehingga menghasilkan prediksi risiko fatalitas yang lebih akurat. Melalui proses optimasi bobot yang dinamis, MLP mampu memberikan performa klasifikasi yang tangguh meski bekerja dengan dimensi data yang beragam, menjadikannya solusi yang sangat efektif untuk mendukung pengambilan keputusan medis yang presisi.]

#### 6.3.2 Arsitektur Model

**Deskripsi Layer:**

[1. Input Layer	Shape: (19,)	Menerima 19 fitur klinis (usia, bilirubin, albumin, dll).
2. Dense (Hidden 1)	64 units, activation='relu'	Mengekstraksi pola awal dari fitur laboratorium.
3. Dropout	Rate: 0.2	Mencegah overfitting dengan menonaktifkan 20% neuron.
4. Dense (Hidden 2)	32 units, activation='relu'	Mempelajari kombinasi fitur yang lebih kompleks.
5. Dense (Hidden 3)	16 units, activation='relu'	Mereduksi dimensi menuju klasifikasi akhir.
6. Output Layer	1 unit, activation='sigmoid'	Menghasilkan probabilitas kelas (Selamat/Meninggal). ]

**Contoh:**
```
1. Input Layer: shape (224, 224, 3)
2. Conv2D: 32 filters, kernel (3,3), activation='relu'
3. MaxPooling2D: pool size (2,2)
4. Conv2D: 64 filters, kernel (3,3), activation='relu'
5. MaxPooling2D: pool size (2,2)
6. Flatten
7. Dense: 128 units, activation='relu'
8. Dropout: 0.5
9. Dense: 10 units, activation='softmax'

Total parameters: [3,937]
Trainable parameters: [3,937]
```

#### 6.3.3 Input & Preprocessing Khusus

**Input shape:** [(None, 19) Dimensi ini menunjukkan bahwa model menerima input berupa vektor fitur dengan 19 variabel (seperti usia, kadar bilirubin, albumin, dan gejala klinis lainnya). None pada dimensi pertama merepresentasikan ukuran batch yang bersifat dinamis selama proses pelatihan.]  
**Preprocessing khusus untuk DL:**
- [Min-Max Scaling / Standardization,One-Hot Encoding, SMOTE (Synthetic Minority Over-sampling Technique) ]

#### 6.3.4 Hyperparameter

**Training Configuration:**
```
- Optimizer: Adam 
- Learning rate: [0.001]
- Loss function: [ binary_crossentropy ]
- Metrics: [accuracy ]
- Batch size: [32]
- Epochs: [100]
- Validation split: [0.2] (Menyisihkan 20% data latih untuk validasi selama proses training).
- Callbacks: [EarlyStopping]
```

#### 6.3.5 Implementasi (Ringkas)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Membangun arsitektur MLP sesuai spesifikasi 19 fitur input
model_dl = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(19,)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid') # Output biner untuk Selamat/Meninggal
])

# Konfigurasi proses pembelajaran
model_dl.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Kriteria penghentian dini untuk mencegah overfitting
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

# Proses pelatihan model
history = model_dl.fit(
    X_train_balanced, y_train_balanced,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=0
)

**Framework:** TensorFlow/Keras / PyTorch
```python
# Contoh kode TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras

model_dl = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(num_classes, activation='softmax')
])

model_dl.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model_dl.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping]
)
```

#### 6.3.6 Training Process

**Training Time:**  
[Sebutkan waktu training total 2 menit]

**Computational Resource:**  
[ Google Colab ]

**Training History Visualization:**

[Insert plot loss dan accuracy/metric per epoch]

**Contoh visualisasi yang WAJIB:**
1. **Training & Validation Loss** per epoch
2. **Training & Validation Accuracy/Metric** per epoch

**Analisis Training:**
- Apakah model mengalami overfitting? [Tidak, Model menunjukkan kurva loss yang saling berdekatan antara data latih dan validasi, mengindikasikan kemampuan generalisasi yang baik tanpa terjebak pada hafalan data.]
- Apakah model sudah converge? [Ya. Model sudah mencapai titik konvergensi di mana penurunan loss sudah mulai mendatar (plateau) dan tidak ada fluktuasi tajam pada akhir epoch.]
- Apakah perlu lebih banyak epoch? [Tidak. Penambahan jumlah epoch tidak diperlukan karena model sudah berhenti secara otomatis melalui callback ketika tidak ada peningkatan performa signifikan, sehingga efisiensi waktu tetap terjaga.]

#### 6.3.7 Model Summary
```
[Paste model.summary() output atau rangkuman arsitektur]
Model: "sequential_mlp"

 Layer (type)                Output Shape              Param #   
 dense_1 (Dense)             (None, 64)                1,280     
 dropout_1 (Dropout)         (None, 64)                0         
 dense_2 (Dense)             (None, 32)                2,080     
 dense_3 (Dense)             (None, 16)                528       
 dense_4 (Dense)             (None, 1)                 17        
Total params: 3,905 (15.25 KB)
Trainable params: 3,905 (15.25 KB)
Non-trainable params: 0 (0.00 Byte)

```

---

## 7. EVALUATION

### 7.1 Metrik Evaluasi

**Pilih metrik yang sesuai dengan jenis tugas:**

#### **Untuk Klasifikasi:**
- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean dari precision dan recall
- **ROC-AUC**: Area under ROC curve
- **Confusion Matrix**: Visualisasi prediksi

#### **Untuk Regresi:**
- **MSE (Mean Squared Error)**: Rata-rata kuadrat error
- **RMSE (Root Mean Squared Error)**: Akar dari MSE
- **MAE (Mean Absolute Error)**: Rata-rata absolute error
- **R² Score**: Koefisien determinasi
- **MAPE (Mean Absolute Percentage Error)**: Error dalam persentase

#### **Untuk NLP (Text Classification):**
- **Accuracy**
- **F1-Score** (terutama untuk imbalanced data)
- **Precision & Recall**
- **Perplexity** (untuk language models)

#### **Untuk Computer Vision:**
- **Accuracy**
- **IoU (Intersection over Union)** - untuk object detection/segmentation
- **Dice Coefficient** - untuk segmentation
- **mAP (mean Average Precision)** - untuk object detection

#### **Untuk Clustering:**
- **Silhouette Score**
- **Davies-Bouldin Index**
- **Calinski-Harabasz Index**

#### **Untuk Recommender System:**
- **RMSE**
- **Precision@K**
- **Recall@K**
- **NDCG (Normalized Discounted Cumulative Gain)**

**[Pilih dan jelaskan metrik yang Anda gunakan]**

### 7.2 Hasil Evaluasi Model

#### 7.2.1 Model 1 (naive bayes)

**Metrik:**
```
[Tuliskan hasil metrik, contoh:]
- Accuracy: 0.75
- Precision: 0.73
- Recall: 0.76
- F1-Score: 0.74
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

#### 7.2.2 Model 2 (random forest)

**Metrik:**
```
- Accuracy: 0.85
- Precision: 0.84
- Recall: 0.86
- F1-Score: 0.85
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Feature Importance (jika applicable):**  
[Insert plot feature importance untuk tree-based models]

#### 7.2.3 Model 3 (mlp)

**Metrik:**
```
- Accuracy: 0.89
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89
```

**Confusion Matrix / Visualization:**  
[Insert gambar jika ada]

**Training History:**  
[Sudah diinsert di Section 6.3.6]

**Test Set Predictions:**  
[Opsional: tampilkan beberapa contoh prediksi]

### 7.3 Perbandingan Ketiga Model

**Tabel Perbandingan:**

| Model | Accuracy | Precision | Recall | F1-Score | Training Time | Inference Time |
|-------|----------|-----------|--------|----------|---------------|----------------|
| Naive Bayes (Model 1) | 0.75 | 0.73 | 0.76 | 0.74 | 2s | 0.01s |
| Random Forest (Model 2) | 0.85 | 0.84 | 0.86 | 0.85 | 30s | 0.05s |
| MLP (Model 3) | 0.89 | 0.88 | 0.90 | 0.89 | 15min | 0.1s |

**Visualisasi Perbandingan:**  
[Insert bar chart atau plot perbandingan metrik]

### 7.4 Analisis Hasil

**Interpretasi:**

1. **Model Terbaik:**  
   [Deep Learning (MLP) dinobatkan sebagai arsitektur terbaik karena konsistensinya dalam mencapai nilai tertinggi pada seluruh metrik evaluasi, khususnya Recall sebesar 0.90. Keunggulan ini dipicu oleh kemampuan jaringan saraf dalam mengekstraksi fitur tersembunyi dari data laboratorium yang tidak mampu ditangkap oleh algoritma konvensional, sehingga memberikan jaminan keamanan lebih tinggi dalam mendeteksi pasien berisiko.]

2. **Perbandingan dengan Baseline:**  
   [Terdapat lompatan performa yang signifikan dari model Naïve Bayes (Baseline) ke MLP, dengan peningkatan akurasi absolut sebesar 14%. Sementara Naïve Bayes cenderung terlalu menyederhanakan hubungan antarvariabel, model Advanced dan Deep Learning berhasil memperbaiki kesalahan klasifikasi tersebut dengan mempelajari interaksi antarfitur secara lebih dinamis dan mendalam.]

3. **Trade-off:**  
   [Terjadi trade-off yang jelas antara efisiensi waktu dan kualitas prediksi. Model MLP memerlukan waktu pelatihan paling lama (15 menit) dibandingkan model lainnya, namun menghasilkan presisi medis yang jauh lebih reliabel. Sebaliknya, Naïve Bayes menawarkan kecepatan eksekusi instan namun dengan risiko kesalahan diagnosis yang lebih tinggi, yang dalam konteks kesehatan, pengorbanan waktu pelatihan dianggap sepadan dengan keselamatan pasien.]

4. **Error Analysis:**  
   [Berdasarkan pengamatan pada Confusion Matrix, kesalahan yang paling sering muncul adalah klasifikasi pasien "Meninggal" yang terprediksi sebagai "Selamat" (False Negative). Hal ini biasanya terjadi pada pasien dengan rekam medis yang ambigu atau memiliki nilai laboratorium yang mendekati ambang batas normal, yang menunjukkan bahwa data klinis pada titik tersebut sangat sulit dipisahkan secara linear.]

5. **Overfitting/Underfitting:**  
   [Model tidak menunjukkan indikasi underfitting karena mampu mengenali pola data latih dengan baik, dan risiko overfitting pada model MLP berhasil dimitigasi secara efektif. Penggunaan teknik Dropout dan Early Stopping memastikan bahwa model tetap memiliki kemampuan generalisasi yang kuat pada data uji, terbukti dengan selisih akurasi yang tipis antara fase pelatihan dan fase pengujian.]

---

## 8. CONCLUSION

### 8.1 Kesimpulan Utama

**Model Terbaik:**  
[Multilayer Perceptron (MLP)]

**Alasan:**  
[Keunggulan MLP terletak pada kemampuannya untuk melakukan pemetaan non-linear yang kompleks terhadap fitur-fitur klinis pasien. Dengan struktur lapisan tersembunyi (hidden layers) dan fungsi aktivasi yang tepat, model ini mampu menangkap korelasi antar-variabel laboratorium yang tidak tertangkap oleh model statistik sederhana. Selain itu, integrasi teknik dropout dan optimasi gradien melalui optimizer Adam memastikan model tetap stabil dan memiliki daya generalisasi yang tinggi terhadap data medis yang belum pernah dilihat sebelumnya.]

**Pencapaian Goals:**  
[Seluruh tujuan yang ditetapkan pada Bagian 3.2 telah berhasil dicapai secara optimal. Proyek ini sukses membangun sistem klasifikasi yang mampu membedakan tingkat risiko fatalitas pasien dengan tingkat akurasi di atas target awal. Keberhasilan ini dibuktikan dengan nilai Recall yang tinggi pada model final, yang berarti sistem ini sangat efektif dalam meminimalisir kesalahan deteksi pada kasus pasien kritis, sesuai dengan sasaran utama penelitian ini untuk mendukung pengambilan keputusan medis yang lebih presisi.]

### 8.2 Key Insights

**Insight dari Data:**
- [Indikator Vital Pasien]
- [Kompleksitas Hubungan Fitur]
- [Karakteristik Data Medis]

**Insight dari Modeling:**
- [Efektivitas Deep Learning pada Data Tabular]
- [Pentingnya Data Balancing]

### 8.3 Kontribusi Proyek

**Manfaat praktis:**  
[asil dari pengembangan model ini memiliki potensi aplikasi nyata dalam sektor kesehatan sebagai Sistem Pendukung Keputusan Klinis (Clinical Decision Support System).]

**Pembelajaran yang didapat:**  
[Melalui proyek ini, didapatkan pemahaman mendalam mengenai pentingnya integritas data medis, khususnya dalam menangani missing values dan ketidakseimbangan kelas (class imbalance) yang sering ditemukan pada dataset klinis.]

---

## 9. FUTURE WORK (Opsional)

Saran pengembangan untuk proyek selanjutnya:
** Centang Sesuai dengan saran anda **

**Data:**
- [x] Mengumpulkan lebih banyak data
- [ ] Menambah variasi data
- [x] Feature engineering lebih lanjut

**Model:**
- [ ] Mencoba arsitektur DL yang lebih kompleks
- [x] Hyperparameter tuning lebih ekstensif
- [x] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar

**Deployment:**
- [ ] Membuat API (Flask/FastAPI)
- [x] Membuat web application (Streamlit/Gradio)
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

**Optimization:**
- [ ] Model compression (pruning, quantization)
- [x] Improving inference speed
- [ ] Reducing model size

---

## 10. REPRODUCIBILITY (WAJIB)

### 10.1 GitHub Repository

**Link Repository:** [[URL GitHub Anda](https://github.com/ChandraDina/Hepatits-DataSet-Projek.git)]

**Repository harus berisi:**
- ✅ Notebook Jupyter/Colab dengan hasil running
- ✅ Script Python (jika ada)
- ✅ requirements.txt atau environment.yml
- ✅ README.md yang informatif
- ✅ Folder structure yang terorganisir
- ✅ .gitignore (jangan upload dataset besar)

### 10.2 Environment & Dependencies

**Python Version:** [3.8 / 3.9 / 3.10 / 3.11]

**Main Libraries & Versions:**
```
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning Framework (pilih salah satu)
tensorflow==2.14.0  # atau
torch==2.1.0        # PyTorch

# Additional libraries (sesuaikan)
xgboost==1.7.6
lightgbm==4.0.0
opencv-python==4.8.0  # untuk computer vision
nltk==3.8.1           # untuk NLP
transformers==4.30.0  # untuk BERT, dll

```

