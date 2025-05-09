# **Laporan Proyek Machine Learning - Rizal Gibran Aldrin P**

## 📌 Project Overview
Dalam era digital saat ini, pertumbuhan e-commerce telah menciptakan tantangan baru bagi pengguna dalam memilih produk yang sesuai dengan kebutuhan dan preferensinya. Salah satu tantangan utama adalah membanjirnya pilihan produk, khususnya di industri smartphone yang sangat kompetitif. Konsumen sering kali merasa kewalahan dengan banyaknya model dan spesifikasi yang tersedia di pasaran.
Sistem rekomendasi telah terbukti menjadi solusi efektif untuk mengurangi kompleksitas dalam pengambilan keputusan pembelian. Menurut Ricci et al. (2015), sistem rekomendasi membantu pengguna menemukan item yang relevan secara personal dalam jumlah besar informasi, dan telah menjadi fitur inti dalam platform seperti Amazon, Netflix, dan Tokopedia [1].

Secara umum, terdapat dua pendekatan utama dalam membangun sistem rekomendasi:
* **Content-Based Filtering**, yang merekomendasikan item serupa dengan item yang telah disukai pengguna berdasarkan deskripsi atau fitur produk.
* **Collaborative Filtering**, yang merekomendasikan item berdasarkan kesamaan preferensi antar pengguna.

Dalam proyek ini, kedua pendekatan tersebut diterapkan pada data yang berisi informasi pengguna, model HP, dan rating pengguna terhadap berbagai model HP. Dengan membangun sistem rekomendasi berbasis data ini, pengguna akan mendapatkan saran produk yang lebih sesuai dan efisien dalam proses pencarian mereka.

**Mengapa Masalah Ini Penting?**

Menurut laporan Statista (2023), pengguna e-commerce di Indonesia telah mencapai lebih dari 180 juta, dengan industri smartphone menjadi salah satu kategori yang paling banyak dicari [2]. Di sisi lain, laporan Nielsen menyatakan bahwa 78% konsumen merasa terbantu dengan rekomendasi produk dalam proses belanja online [3]. Oleh karena itu, membangun sistem rekomendasi dalam konteks produk smartphone bukan hanya relevan, tetapi juga sangat bermanfaat dalam meningkatkan kepuasan pelanggan dan konversi penjualan.

**Referensi**

[1] Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender Systems Handbook. Springer.

[2] Statista. (2023). Number of digital buyers in Indonesia from 2017 to 2028. Retrieved from https://www.statista.com

[3] Nielsen. (2020). The Power of Personalization: Offering the Right Recommendations to Consumers. Retrieved from https://www.nielsen.com


## 💼 Business Understanding

### 🧠 Problem Statements
1. Bagaimana membangun sistem rekomendasi produk ponsel yang dapat memberikan saran relevan berdasarkan deskripsi produk?
2. Bagaimana memanfaatkan interaksi pengguna berupa rating untuk menghasilkan rekomendasi yang akurat?
3. Bagaimana membandingkan performa pendekatan content-based dan collaborative filtering dalam konteks rekomendasi produk?

### 🎯 Goals
1. Membangun sistem rekomendasi berbasis content-based filtering menggunakan TF-IDF dan cosine similarity untuk mengukur kesamaan antar produk.
2. Mengembangkan sistem rekomendasi berbasis collaborative filtering menggunakan pendekatan Neural Collaborative Filtering dengan model embedding.
3. Mengevaluasi performa kedua pendekatan untuk mengetahui kekuatan dan kelemahan masing-masing.

### 💡 Solution Statements
Dalam menyelesaikan permasalahan sistem rekomendasi ini, dua pendekatan utama digunakan:

1. Pendekatan 1: Content-Based Filtering

   Pendekatan ini memanfaatkan informasi deskriptif dari produk, dalam hal ini nama atau tipe ponsel, untuk merekomendasikan produk yang mirip dengan produk yang disukai pengguna. Teknik yang digunakan antara lain:
   * TF-IDF Vectorization: Teknik ini digunakan untuk mengubah teks produk menjadi representasi numerik berdasarkan frekuensi kata dan inverse document frequency.
   * Cosine Similarity: Digunakan untuk menghitung tingkat kemiripan antara dua produk berdasarkan representasi TF-IDF mereka.
   * Rekomendasi diberikan berdasarkan produk yang memiliki kemiripan tertinggi terhadap produk yang sebelumnya diberi rating tinggi oleh pengguna.

2. Pendekatan 2: Collaborative Filtering (Neural Collaborative Filtering)

   Pendekatan ini menggunakan data interaksi pengguna dan produk dalam bentuk rating. Model yang digunakan adalah:
   * RecommenderNet: Sebuah neural network yang terdiri dari embedding layer untuk representasi user dan produk, serta bias untuk masing-masing.
   * Model dibangun dengan Keras dan TensorFlow, dan dilatih untuk meminimalkan Root Mean Squared Error (RMSE).
   * Pendekatan ini memungkinkan model untuk mempelajari preferensi pengguna secara tidak langsung melalui pola rating yang tersedia.

## 📁 Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Cellphones Recommendation Dataset" dari [Kaggle](https://www.kaggle.com/datasets/meirnizri/cellphones-recommendations). Dataset ini terdiri dari tiga file utama:

1. Dataset 1: cellphones data.csv

   Berisi data tentang ponsel terpopuler di AS pada tahun 2022, dengan variabel-variabel :
    * cellphone_id: ID unik dari ponsel.
    * model: Nama model ponsel.
    * brand: Merek dari ponsel.
    * operating system: Sistem operasi yang digunakan ponsel (iOS/Android).
    * internal memory: Penyimpanan internal ponsel.
    * RAM: RAM ponsel.
    * performance: Nilai kinerja ponsel berdasarkan nilai AnTuTu.
    * main camera: Besaran piksel pada kamera utama atau kamera belakang.
    * selfie camera: besaran piksel pada kamera depan.
    * battery size: Kapasitas baterai (mAh).
    * screen size: Lebar layar ponsel (inch).
    * weight: Berat dari ponsel (gram).
    * price: Harga dari ponsel (USD).
    * release date: Tanggal keluar atau rilis pertama ponsel.

2. Datset 2: cellphones ratings.csv

   Berisi data rating yang diberikan pengguna terhadap model ponsel, dengan variabel-variabel :
    * user_id: ID unik pengguna
    * model: Nama model ponsel
    * rating: Nilai rating (skala 1–10)

3. Dataset 3: cellphones users.csv

   Berisi informasi tambahan tentang pengguna, dengan variabel-variabel :
   * user_id : ID unik pengguna
   * Age: Umur pengguna ponsel
   * Gender: Jenis kelamin pengguna ponsel
   * Occupation: Pekerjaan pengguna ponsel

Untuk memahami data, dilakukan beberapa tahapan yang diperlukan, yaitu:
1. Cek dimensi data
   
   * Pada dataset 1 terdapat 33 jumlah baris data, dengan 14 kolom
     *  8 kolom bertipe int64 : cellphone_id, internal memory, RAM, main camera, selfie camera, battery size, weight, price
     *  4 kolom bertipe object: brand, model, operatinf system, release date
     *  2 kolom bertipe float64 : performance, screen size
   * Pada dataset 2 terdapat 990 jumlah baris data, dengan 3 kolom
     * 3 kolom bertipe int64: user_id, model, rating
   * Pada dataset 3 terdapat 99 jumlah baris data, dengan 4 kolom
     * 2 kolom bertipe int64: user_id, age
     * 2 kolom bertipe object: gender, occupation

2. Cek kondisi data


## 🧪 Data Preparation

1. Penggabungan dan Penyelarasan Data

   Ketiga data yaitu cellphones data.csv, cellphones ratings.csv, dan cellphones users.csv, digabungkan berdasarkan kolom model dan user_id untuk menghasilkan satu dataset utama (full_df) yang digunakan untuk kedua pendekatan sistem rekomendasi

2. Penggabungan Data Rating dan Pengguna

   Dataset cellphones ratings juga di-join dengan data pengguna dari cellphones users berdasarkan user_id untuk memperkaya informasi seperti usia dan jenis kelamin pengguna.

2. Agregasi Rating

   Karena satu model ponsel bisa memiliki banyak rating, dilakukan agregasi dengan cara menghitung rata-rata rating untuk setiap model. Hasil ini digunakan pada pendekatan Content-Based Filtering agar tiap produk memiliki representasi skor tunggal yang lebih stabil.

4. Pembersihan Duplikasi

   Diperiksa dan dihapus kemungkinan duplikasi data model ponsel dalam dataset setelah proses join dan agregasi.

5. Persiapan Matriks Rating

   Untuk pendekatan Collaborative Filtering, dibuat matriks user-item dari data rating, di mana baris merepresentasikan pengguna dan kolom merepresentasikan model ponsel. Nilai dalam matriks adalah rating dari pengguna terhadap produk tersebut.


## 🛠️ Modeling

Dalam tahap ini, dua pendekatan sistem rekomendasi diterapkan untuk menyelesaikan permasalahan: Content-Based Filtering dan Collaborative Filtering. Setiap pendekatan menggunakan algoritma yang berbeda dan menghasilkan output rekomendasi dalam bentuk Top-N recommendation untuk pengguna atau produk.

**1. Content-Based Filtering**

Pada pendekatan ini, sistem merekomendasikan produk berdasarkan kemiripan antar produk, tanpa mempertimbangkan perilaku pengguna lain. Fitur produk diproses menggunakan metode TF-IDF (Term Frequency–Inverse Document Frequency) yang diambil dari atribut description atau fitur-fitur tekstual lainnya.
Kemudian, digunakan cosine similarity untuk menghitung tingkat kemiripan antar produk. Model ini akan memberikan rekomendasi produk yang paling mirip berdasarkan model ponsel yang telah dilihat atau disukai pengguna.

Kelebihan:
* Tidak membutuhkan data rating pengguna lain (tidak rentan terhadap cold-start user).
* Cocok untuk merekomendasikan item baru yang belum memiliki banyak rating.

Kekurangan:
* Hanya bisa merekomendasikan item yang mirip dengan yang sudah dikenal.
* Tidak mempertimbangkan selera atau preferensi pengguna lain.

**2. Collaborative Filtering (Neural Collaborative Filtering)**
Pendekatan ini menggunakan teknik Neural Collaborative Filtering (NCF) yang dibangun dengan arsitektur neural network. Model ini belajar dari interaksi antara pengguna dan item (rating) untuk memprediksi rating yang mungkin diberikan pengguna terhadap produk tertentu, lalu merekomendasikan produk dengan prediksi rating tertinggi.

Model NCF dilatih menggunakan embedding layer untuk pengguna dan item, digabungkan dengan beberapa dense layer, dan dioptimasi menggunakan fungsi loss `root_mean_squared_error`.

Kelebihan:
* Dapat menangkap pola kompleks antara pengguna dan produk.
* Mampu memberikan rekomendasi personalisasi berdasarkan riwayat interaksi pengguna.

Kekurangan:
* Butuh jumlah data rating yang cukup untuk performa optimal.
* Rentan terhadap masalah cold-start, terutama pengguna atau item baru.

**Output: Top-N Recommendation**

Setelah kedua model selesai dibangun, sistem dapat memberikan Top-5 rekomendasi produk:
* Content-Based: Rekomendasi 5 produk paling mirip berdasarkan deskripsi atau fitur dari model ponsel yang dipilih pengguna.
* Collaborative Filtering: Rekomendasi 5 produk dengan rating tertinggi yang diprediksi untuk pengguna tertentu berdasarkan interaksi pengguna lainnya.


## 📊 Evaluation

Metrik evaluasi yang digunakan untuk mengevaluasi performa kedua pendekatan sistem rekomendasi, digunakan metrik yang sesuai dengan masing-masing metode:

1. **Content-Based Filtering – TF-IDF & Cosine Similarity**

Metrik Evaluasi:
  * Cosine Similarity : digunakan untuk mengukur kemiripan antara produk berdasarkan fitur teks. Metrik ini mengukur sudut antara dua vektor representasi produk (semakin kecil sudut, semakin mirip).

Formula Cosine Similarity:
![Formula Cosine Similarity](/content/drive/MyDrive/Colab Notebooks/rumus cosine.png)

* A dan 𝐵: vektor representasi dari dua produk
* Nilai cosine similarity berkisar antara 0 hingga 1, di mana nilai mendekati 1 menunjukkan kemiripan tinggi.

Hasil Evaluasi:

Evaluasi dilakukan secara kualitatif dengan memeriksa apakah produk yang direkomendasikan benar-benar relevan atau serupa dengan produk favorit pengguna. Hasil menunjukkan bahwa sistem berhasil merekomendasikan produk yang sejenis dari sisi tipe dan merek.

2. Collaborative Filtering – Neural Collaborative Filtering
  Metrik Evaluasi:
   * Root Mean Squared Error (RMSE) : RMSE digunakan untuk mengukur perbedaan antara nilai rating aktual dan prediksi yang dihasilkan oleh model. Metrik ini cocok untuk regresi karena mempertimbangkan besar kesalahan (error) dalam satuan yang sama dengan target aslinya (rating).

Formula RMSE:

![rumus rmse](https://github.com/user-attachments/assets/82978cad-dc32-48a3-b505-981f612e0654)

Hasil Evaluasi:

Dari grafik RMSE selama 100 epoch, diperoleh tren penurunan error yang cukup baik pada data pelatihan dan data validasi, walaupun terdapat sedikit overfitting di akhir training. Pada akhir pelatihan:
* RMSE Data Training: ~0.10
* RMSE Data Validasi: ~0.16

Nilai RMSE yang rendah ini menunjukkan bahwa model mampu memprediksi rating pengguna terhadap produk dengan akurasi yang cukup tinggi.


## ✅ Kesimpulan
Pada proyek ini, telah dibangun dan dibandingkan dua pendekatan sistem rekomendasi berbasis data:
1. Content-Based Filtering menggunakan teknik TF-IDF dan cosine similarity untuk merekomendasikan produk ponsel berdasarkan kemiripan fitur deskriptif produk.
2. Collaborative Filtering menggunakan pendekatan Neural Collaborative Filtering (RecommenderNet) dengan pemodelan embedding pengguna dan produk, untuk memprediksi interaksi berdasarkan histori.

Berdasarkan hasil evaluasi:
* Pendekatan Collaborative Filtering menunjukkan performa yang lebih baik secara kuantitatif dengan RMSE 0.17–0.19, mencerminkan akurasi yang baik dalam memprediksi preferensi pengguna.
* Pendekatan Content-Based Filtering tetap memiliki keunggulan dalam mengatasi masalah cold-start untuk produk baru, serta tidak memerlukan histori interaksi dari pengguna.

Insight
* Kombinasi kedua pendekatan dalam bentuk hybrid recommendation system berpotensi memberikan performa yang lebih seimbang antara akurasi dan cakupan rekomendasi.
* Penambahan informasi lebih lanjut seperti kategori produk, harga, atau ulasan pengguna bisa meningkatkan kualitas sistem rekomendasi di masa depan.
