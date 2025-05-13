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
* Data tidak terdapat duplikat pada ketiga dataset.
* Dataset 1 dan 2 tidak terdapat missing value, dataset 3 terdapat missing value pada kolom occupation.

3. Melakukan Exploratory Data Analysis (EDA)
* Terdapat 10 brand yaitu ['Apple' 'Asus' 'Samsung' 'Google' 'OnePlus' 'Oppo' 'Vivo' 'Xiaomi' 'Sony'
 'Motorola']. Model paling paling banyak berasal dari brand Samsung 8 buah, kemudian diikuti Apple 6 buah dan ada Motorola, OnePlus serta Xiaomi dengan jumlah yang sama 4 buah
* Pada tabel rating, nilai terendah 1 dan nilai terbesar 18. Nilai rating 18 pada dataset ponsel mengindikasikan adanya anomali atau kesalahan, mengingat skala rating yang umum digunakan adalah 1-10 atau 1-5.
* Pada kolom gender terdapat nilai `select gender` yang seharusnya hanya 2 gender yaitu female dan male, kemungkinan user memilih tidak menyebutkan gendernya. Tetapi pada proyek kali ini variabel tersebut tidak dipergunakan.

## 🧪 Data Preparation

1. Penggabungan dan Penyelarasan Data

   Ketiga data yaitu cellphones `data.csv`, `cellphones ratings.csv`, dan `cellphones users.csv`, digabungkan berdasarkan kolom `model` dan `user_id` untuk menghasilkan satu dataset utama (`full_df`), yang digunakan untuk kedua pendekatan sistem rekomendasi. Dataset cellphones ratings juga di-join dengan data pengguna dari cellphones users berdasarkan `user_id`.

2. Mengubah kolom 'release date' yang sebelumnya object menjadi datetime.
   
4. Menangani anomali nilai rating

   Terdapat nilai rating 18 pada dataset ponsel mengindikasikan adanya anomali atau kesalahan, mengingat skala rating yang umum digunakan adalah 1-10 atau 1-5. Untuk mengatasi hal tersebut, dilakukan perhitungan rata-rata rating dari model terkait, lalu menggantikan anomali rating dengan nilai rata-rata tersebut.

5. Menangani karakter non-breaking space

   Non-breaking space (karakter \xa0 dalam representasi Python) adalah karakter spasi khusus yang mencegah pemisahan baris otomatis pada posisi spasi tersebut. Karakter ini sering digunakan dalam tipografi untuk menjaga agar elemen-elemen tertentu tetap bersama, seperti angka dan satuannya ("10 kg"). Untuk mengatasinya perlu membersihkan data dengan menghapus karakter \xa0 dari nama model 'Pixel 6 Pro\xa0', ada dua cara:
   * Mengganti karakter \xa0 dengan spasi biasa -> `full_df['model'] = full_df['model'].str.replace(u'\xa0', u' ')`
   * Menghapus semua karakter whitespace (termasuk \xa0) di awal dan akhir string -> `full_df['model'] = full_df['model'].str.strip()`
   * Yang digunakan pada proyek ini yaitu cara kedua.  Ini bisa berguna jika ada spasi atau masalah pemformatan lain yang tidak diinginkan dalam data.

Setelah melakukan data preparation secara umum diatas, selanjutnya dilakukan `.copy()` digunakan untuk membuat salilnan data sebelum dilakukan data preparation masing-masing model, yang bertujuan agar dataset asli tetap aman. Kemudian mengurutkan data salinan berdasarkan ID ponsel.

### Content-Based Filtering

Untuk membangun sistem rekomendasi berbasis konten (Content-Based Filtering), sebelumnya data utama (`full_df`) disalin kemudian data disiapkan melalui beberapa tahapan berikut:
1. Menghitung rata-rata rating per model: Langkah ini menghitung rata-rata rating dari masing-masing produk (`model`), agar dapat digunakan sebagai salah satu informasi relevansi produk dalam hasil rekomendasi.
2. Menggabungkan informasi rating dengan data produk model: Menggabungkan data rating rata-rata ke dalam dataset utama (`content_df` - salinan `ull_df`) berdasarkan kolom model.
3. Menghapus duplikasi produk: Beberapa produk mungkin muncul lebih dari sekali dalam dataset. Maka, duplikat dihapus berdasarkan kolom model untuk memastikan tiap produk hanya satu kali muncul.
4. Membuat fitur gabungan (Combined Features): Fitur gabungan dibentuk dengan menggabungkan kolom-kolom teks dan numerik (yang dikonversi ke `string`) menjadi satu string deskripsi panjang untuk masing-masing produk. Ini digunakan sebagai dasar untuk analisis kesamaan antarproduk. Kolom yang digunakan `['model', 'operating system', 'internal memory', 'RAM', 'performance', 'main camera', 'battery size', 'screen size', 'price']`.
5. Preprocessing dengan Lemmatization: Setiap deskripsi gabungan difilter dengan proses lemmatization, yaitu mengubah kata ke bentuk dasarnya. Ini membantu dalam generalisasi kata-kata saat membuat representasi teks.
6. Ekstraksi fitur dengan TF-IDF dan penghitungan Cosine Similarity : Menggunakan **TF-IDF Vectorizer** untuk mengubah teks gabungan menjadi vektor numerik. Parameter yang digunakan yaitu `max_df`, `min_df` dan `ngram_range`. Kemudian, digunakan **cosine similarity** untuk mengukur kemiripan antarproduk berdasarkan representasi vektor **TF-IDF**.
   * `TfidfVectorizer` digunakan untuk mengubah teks menjadi representasi numerik yang dapat dipahami oleh model machine learning. Ia melakukan ini menggunakan teknik yang disebut TF-IDF (Term Frequency-Inverse Document Frequency). Sederhananya, ini membantu menentukan pentingnya kata dalam sebuah dokumen relatif terhadap kumpulan dokumen.
   * `cosine_similarity` adalah cara untuk mengukur seberapa mirip dua teks (atau data apa pun yang direpresentasikan sebagai vektor) satu sama lain. Ini didasarkan pada sudut antara vektor yang mewakili teks. Kesamaan kosinus 1 berarti teksnya identik, 0 berarti sama sekali berbeda, dan nilai di antaranya menunjukkan tingkat kesamaan yang bervariasi.
   * `ngram_range=(1,3)` digunakan agar model mempertimbangkan unigram, bigram, dan trigram.
   * `max_df=0.8` dan `min_df=0.15` berfungsi sebagai filter: hanya mempertahankan kata-kata yang tidak terlalu umum atau terlalu jarang.

### Collaborative Filterring

Untuk pendekatan Collaborative Filtering, data disiapkan melalui langkah-langkah sebagai berikut:
1. Persiapan dataset : Data `rating` digunakan sebagai sumber utama yang berisi informasi interaksi antara pengguna (`user_id`) dan produk (`cellphone_id`) beserta nilai `rating`.
2. Menghitung jumlah user dan produk: Menampilkan jumlah pengguna unik dan jumlah model ponsel unik yang terdapat dalam dataset.
3. Normalisasi ID : Untuk memudahkan pemrosesan oleh model, ID pengguna dan produk diubah menjadi indeks bilangan bulat berurutan (mulai dari 0), sehingga lebih efisien saat digunakan dalam model Neural Collaborative Filtering.
4. Konversi tipe data rating : Kolom `rating` dikonversi ke dalam format `float32` agar kompatibel dengan TensorFlow/Keras saat pelatihan model.
5. Normalisasi skala rating : Rating dinormalisasi ke skala 0–1 menggunakan formula:

   rating_normalized = (rating - min) / (max - min)
​
6. Membuat data input (x) dan target (y): Data input `x` terdiri dari pasangan (`user_id`, `cellphone_id`), dan `y` adalah target rating hasil normalisasi.
7. Split dataset : Dataset dibagi menjadi Train dan Validation, data dibagi menjadi 80% untuk pelatihan dan 20% untuk validasi. Ini dilakukan agar model dapat dilatih dan dievaluasi tanpa kebocoran data (data leakage).


## 🛠️ Modeling

Dalam tahap ini, dua pendekatan sistem rekomendasi diterapkan untuk menyelesaikan permasalahan: Content-Based Filtering dan Collaborative Filtering. Setiap pendekatan menggunakan algoritma yang berbeda dan menghasilkan output rekomendasi dalam bentuk Top-N recommendation untuk pengguna atau produk.

#### 1. Content-Based Filtering

Pada pendekatan ini, sistem merekomendasikan produk berdasarkan kemiripan antar produk, tanpa mempertimbangkan perilaku pengguna lain. Fitur produk diproses menggunakan metode TF-IDF (Term Frequency–Inverse Document Frequency) yang diambil dari atribut description atau fitur-fitur tekstual lainnya. Kemudian, digunakan cosine similarity untuk menghitung tingkat kemiripan antar produk. Model ini akan memberikan rekomendasi produk yang paling mirip berdasarkan model ponsel yang telah dilihat atau disukai pengguna.

Kelebihan:
* Tidak membutuhkan data rating pengguna lain (tidak rentan terhadap cold-start user).
* Cocok untuk merekomendasikan item baru yang belum memiliki banyak rating.
* Dapat memberikan rekomendasi pada produk baru selama fitur kontennya tersedia.

Kekurangan:
* Hanya bisa merekomendasikan item yang mirip dengan yang sudah dikenal.
* Tidak mempertimbangkan selera atau preferensi pengguna lain.
* Tidak bisa merekomendasikan produk jika semua fiturnya kosong/missing.

### 2. Collaborative Filtering (Neural Collaborative Filtering)

Pendekatan kedua dalam sistem rekomendasi ini adalah Collaborative Filtering berbasis Neural Network (NCF). Berbeda dari content-based yang mengandalkan fitur produk, collaborative filtering menggunakan interaksi historis antara pengguna dan produk. Model ini dibangun menggunakan TensorFlow/Keras dan memanfaatkan teknik embedding untuk merepresentasikan user dan item (ponsel) dalam bentuk vektor laten. 

Tahapan membangun model rekomendasi dengan mendefinisikan sebuah kelas yang disebut `RecommenderNet`, yang merupakan inti dari sistem rekomendasi collaborative filtering. Kelas ini dibangun menggunakan TensorFlow/Keras. Dengan mengambil embeddings dan bias, ia mengambil vektor embedding dan bias untuk pengguna dan ponsel yang diberikan dari lapisan embedding. Kemudian menghitung interaksi antara pengguna dan ponsel menggunakan perkalian dot (`tf.tensordot`). Ini menangkap seberapa besar pengguna mungkin menyukai ponsel tersebut. Selanjutnya penggabungan dengan menambahkan bias pengguna dan bias ponsel ke istilah interaksi. Terakhir menerapkan fungsi aktivasi sigmoid (`tf.nn.sigmoid`) untuk menghasilkan prediksi dalam rentang 0 hingga 1, yang merepresentasikan rating yang diprediksi (setelah normalisasi).

Model ini belajar dari interaksi antara pengguna dan item (`rating`) untuk memprediksi rating yang mungkin diberikan pengguna terhadap produk tertentu, lalu merekomendasikan produk dengan prediksi rating tertinggi. Model NCF dilatih menggunakan embedding layer untuk pengguna dan item, digabungkan dengan beberapa dense layer, dan dioptimasi menggunakan fungsi loss `root_mean_squared_error`.

Kelebihan:
* Mampu menangkap pola kompleks antara user dan item tanpa membutuhkan fitur konten.
* Mampu memberikan rekomendasi personalisasi berdasarkan riwayat interaksi pengguna.

Kekurangan:
* Membutuhkan banyak data interaksi pengguna untuk hasil yang akurat.
* Tidak dapat merekomendasikan produk baru yang belum memiliki interaksi (cold-start problem).

**Output: Top-N Recommendation**

Setelah kedua model selesai dibangun, sistem dapat memberikan Top-5 rekomendasi produk:
* Content-Based: Rekomendasi 5 produk paling mirip berdasarkan deskripsi atau fitur dari model ponsel yang dipilih pengguna.
* Collaborative Filtering: Rekomendasi 5 produk dengan rating tertinggi yang diprediksi untuk pengguna tertentu berdasarkan interaksi pengguna lainnya.

Tambahan, training pada model ini menggunakan `Binary Crossentropy` untuk menghitung `loss function`, `Adam` (Adaptive Moment Estimation) sebagai `optimizer` dengan nilai 0.005, dan `root mean squared error` (RMSE) sebagai metrics evaluation, nilai `batch_size` = 32, dan `epochs` = 100.

### ⭐ Mendapatkan Rekomendasi Ponsel

#### 📱 Content-Based Filtering (Mirip dengan "10 Pro")
![image](https://github.com/user-attachments/assets/662cc3d7-074a-4988-873a-5fd20a08b636)

Menampilkan 5 rekomendasi ponsel berdasarkan kemiripan konten (fitur seperti model/brand) terhadap model "10 Pro, yang bertujuan untuk memberikan rekomendasi ponsel yang mirip dengan satu produk tertentu berdasarkan fitur.

#### 🤖 Collaborative Filtering (User-Based Recommendation)

![image](https://github.com/user-attachments/assets/7d87ef73-974e-4f80-9702-1e3e9147d977)

Menampilkan rekomendasi ponsel untuk user ID tertentu (di sini: user 1). Tujuannya yaitu memberikan rekomendasi berdasarkan kesamaan perilaku pengguna, yaitu berdasarkan rating user serupa.

#### 🧾 Get 10 Phone Recommendations
![image](https://github.com/user-attachments/assets/d3f1dc79-49e3-4ab0-9770-5c34cdc26cc8)

Menampilkan 10 ponsel dengan average rating tertinggi dari seluruh pengguna, dengan metode ranking global berdasarkan rata-rata rating. Bertujuan untuk menunjukkan produk-produk paling disukai oleh semua pengguna secara umum. Ponsel seperti iPhone 13, iPhone 13 Pro, dan iPhone Mini merupakan 3 ponsel dengan rating tertinggi.


## 📈 Evaluation

Metrik evaluasi yang digunakan untuk mengevaluasi performa kedua pendekatan sistem rekomendasi, digunakan metrik yang sesuai dengan masing-masing metode:

### Content-Based Filtering

Untuk mengevaluasi performa sistem rekomendasi berbasis konten, digunakan pendekatan label biner berdasarkan threshold rating. Tujuannya adalah mengukur apakah sistem merekomendasikan item yang relevan (rating tinggi) dan menghindari item yang tidak relevan (rating rendah).

#### Pendekatan Evaluasi

1. Threshold Rating:
   * Digunakan nilai ambang batas `6` dari skala rating 1–10.
   * Rating di atas atau sama dengan 6 dianggap *positif* (relevan).
2. Label Biner:
   * `1` jika rating aktual ≥ 6 (positif), `0` jika tidak.
   * `1` jika produk direkomendasikan oleh sistem, `0` jika tidak.
3. Evaluasi dilakukan untuk satu pengguna tertentu (`user_id = 258`), dengan membandingkan item yang telah dinilai dengan daftar rekomendasi berbasis konten.

#### Metrik evaluasi menggunakan tiga metrik utama:
1. Precision: Proporsi item yang direkomendasikan dan benar-benar relevan.
2. Recall: Proporsi item relevan yang berhasil direkomendasikan.
3. F1-Score: Harmonik rata-rata antara precision dan recall.

#### Hasil Evaluasi
Content-Based Filtering Metrics (with threshold 6):
* Precision: 1.0
* Recall: 0.3333333333333333
* F1-score: 0.5

#### Interpretasi
* Precision = 1.0 berarti semua item yang direkomendasikan oleh sistem benar-benar disukai pengguna.
* Recall = 0.33 berarti hanya sepertiga dari item relevan yang berhasil direkomendasikan oleh sistem.
* F1-score = 0.5 menunjukkan keseimbangan moderat antara precision dan recall.

Dengan kata lain, sistem sangat tepat dalam merekomendasikan (tidak salah merekomendasikan), tetapi belum cukup luas cakupannya (tidak semua item relevan berhasil direkomendasikan).

### Collaborative Filtering – Neural Collaborative Filtering

Model Collaborative Filtering menggunakan pendekatan Matrix Factorization dengan arsitektur embedding berbasis neural network. Evaluasi dilakukan dengan metrik Root Mean Squared Error (RMSE) pada data latih dan validasi selama 100 epoch

Root Mean Squared Error (RMSE) digunakan untuk mengukur perbedaan antara nilai rating aktual dan prediksi yang dihasilkan oleh model. Metrik ini cocok untuk regresi karena mempertimbangkan besar kesalahan (error) dalam satuan yang sama dengan target aslinya (rating).

Formula RMSE:

![rumus rmse](https://github.com/user-attachments/assets/82978cad-dc32-48a3-b505-981f612e0654)

#### Visualisasi Hasil

![image](https://github.com/user-attachments/assets/4c78994f-3baf-4b7d-a34f-20473e2c2dbc)


Gambar di atas menunjukkan grafik perkembangan RMSE terhadap epoch pelatihan.
* Sumbu X: Jumlah epoch (iterasi pelatihan)
* Sumbu Y: Nilai RMSE
* Garis Biru: RMSE pada data train
* Garis Oranye: RMSE pada data test (validasi)

#### Analisis
* Gambar menunjukkan tren penurunan RMSE pada data training dan validasi seiring bertambahnya epoch.
* Model mencapai konvergensi yang stabil di akhir training. Ini mengindikasikan bahwa model tidak mengalami overfitting yang signifikan.
* Tidak terdapat tanda overfitting ekstrem, karena perbedaan RMSE train dan test cukup stabil setelah konvergen.
* Final RMSE on Validation Set: 0.2754, RMSE di bawah 0.3 tergolong sangat baik untuk model rekomendasi berbasis rating prediktif.
* Model mampu mempelajari representasi pengguna dan item secara efisien melalui layer embedding dan meminimalkan error secara konsisten.


## ✅ Conclusion

Berdasarkan hasil evaluasi yang telah dilakukan, kesimpulan perbandingan antara kedua pendekatan sistem rekomendasi dengan pendekatan Content-Based Filtering dan Collaborative Filtering. 
1. Content-Based Filtering sangat selektif, tapi cenderung terlalu berhati-hati sehingga kehilangan beberapa rekomendasi yang relevan. Model ini cocok jika tujuan utamanya adalah menghindari rekomendasi yang salah, tetapi kurang bagus dalam menangkap semua preferensi pengguna. Content-Based Filtering tetap bermanfaat, terutama untuk user/item baru (cold-start), namun dalam hal akurasi prediksi, CF lebih unggul.
2. Collaborative Filtering lebih fleksibel dalam mengenali pola preferensi pengguna, dan hasil prediksi rating-nya relatif akurat. Model ini cocok untuk skenario dengan banyak data interaksi dan ketika konten produk tidak terlalu informatif. Collaborative Filtering menunjukkan performa yang lebih unggul secara metrik RMSE dan kemungkinan besar precision/recall pada top-N juga lebih baik, karena model mampu mempelajari pola rating antar pengguna dan item.
  
Jika ingin rekomendasi yang sangat presisi dan berbasis fitur produk, maka content-based cocok. Tapi untuk sistem yang mampu menangkap preferensi pengguna secara keseluruhan, collaborative filtering menawarkan performa yang lebih seimbang dan fleksibel.
