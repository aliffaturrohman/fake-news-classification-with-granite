# 📰 Membongkar Hoaks dengan Kecerdasan Buatan: Klasifikasi Berita Palsu menggunakan IBM Granite

![Fake News Classification](https://storage.googleapis.com/kaggle-datasets-images/3950844/6875518/662116dc40ef78f6d3a65355233cb215/dataset-cover.JPG?t=2023-11-03-17-46-02)

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![Machine Learning](https://img.shields.io/badge/-Machine%20Learning-orange)](https://en.wikipedia.org/wiki/Machine_learning)
[![NLP](https://img.shields.io/badge/-NLP-yellowgreen)](https://en.wikipedia.org/wiki/Natural_language_processing)

## 🚀 Ringkasan Proyek

Proyek ini bertujuan untuk mengembangkan sistem klasifikasi berita otomatis yang dapat membedakan antara berita `REAL` (nyata) dan `FAKE` (palsu) menggunakan model bahasa besar (LLM) **Granite 3.3:8b** dari IBM. Model ini dijalankan secara lokal melalui **Ollama** dan diintegrasikan dengan **LangChain** di lingkungan Google Colab. Pendekatan utama yang digunakan adalah *few-shot prompting* dan *tuning parameter* LLM yang cermat untuk memastikan respons yang konsisten dan akurat.

## ✨ Fitur Utama

* **Klasifikasi Berita Otomatis:** Mengidentifikasi kategori berita (`REAL` atau `FAKE`) dari judul dan teks artikel.
* **Penggunaan LLM Lokal:** Memanfaatkan IBM Granite 3.3:8b yang berjalan di lingkungan lokal dengan Ollama, menawarkan fleksibilitas dan kontrol lebih.
* **Integrasi LangChain:** Membangun alur kerja LLM yang efisien dan terstruktur.
* **Few-Shot Prompting:** Membimbing model dengan contoh-contoh spesifik untuk memahami kriteria klasifikasi.
* **Tuning Parameter LLM:** Optimalisasi `temperature`, `top_k`, `top_p`, `max_tokens`, dan `min_tokens` untuk output yang deterministik.
* **Evaluasi Kinerja:** Analisis menggunakan Confusion Matrix dan Classification Report untuk mengukur akurasi, presisi, dan *recall* model.

## 🛠️ Teknologi yang Digunakan

* **Python 3.10**
* **Jupyter Notebook** (Google Colab)
* **IBM Granite 3.3:8b** (via Ollama)
* **Ollama**
* **LangChain**
* **Pandas**: Untuk manipulasi dan analisis data.
* **NumPy**: Untuk operasi numerik.
* **Matplotlib** & **Seaborn**: Untuk visualisasi data, khususnya Confusion Matrix.
* **Scikit-learn (sklearn.metrics)**: Untuk metrik evaluasi model seperti Confusion Matrix dan Classification Report.
* **re**: Untuk ekspresi reguler (membersihkan output LLM).

## 📊 Dataset

Dataset yang digunakan dalam proyek ini adalah kumpulan berita yang dikategorikan sebagai `REAL` atau `FAKE`. Untuk tujuan eksperimen cepat, 2000 baris pertama dari dataset digunakan.

Anda dapat mengunduh dataset dari:
[Dataset Fake News di Kaggle](https://www.kaggle.com/datasets/rajatkumar30/fake-news)

## ⚙️ Instalasi dan Penggunaan

Untuk menjalankan proyek ini di lingkungan Google Colab:

1.  **Unduh dan Ekstrak Dataset:**
    ```bash
    !curl -L -o real_fake_news_dataset.zip [https://github.com/aliffaturrohman/fake-news-classification-with-granite/raw/8d8d927d36e171028dd78bb474816e1023ed6d8f/real_fake_news_dataset.zip](https://github.com/aliffaturrohman/fake-news-classification-with-granite/raw/8d8d927d36e171028dd78bb474816e1023ed6d8f/real_fake_news_dataset.zip)
    ```
    (Catatan: Pastikan dataset sudah berada di lokasi yang benar, misalnya `/content/real_fake_news_dataset.zip`).

2.  **Instal Ollama dan LangChain:**
    ```bash
    !pip install colab-xterm -q
    !curl [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh
    !pip install langchain langchain-ollama -q
    !pip install langchain-community langchain-core -q
    ```

3.  **Jalankan Ollama Server dan Unduh Model Granite:**
    Setelah menjalankan sel ini, akan muncul terminal Xterm di Colab. Di terminal tersebut, jalankan perintah berikut secara berurutan:
    ```bash
    ollama serve
    ollama pull granite3.3:8b
    ```
    Tunggu hingga model selesai diunduh. Setelah itu, **jangan tutup terminal Xterm** karena server Ollama harus tetap berjalan.

4.  **Jalankan Kode Notebook:**
    Buka file `Fake_News_Classiffication_with_IBM_Granite.ipynb` dan jalankan setiap sel secara berurutan.

## 💡 Insight dan Rekomendasi

### Insight Utama

Model ini menunjukkan kinerja yang lebih baik dalam mengidentifikasi berita **`FAKE`** (precision 0.90) dibandingkan berita **`REAL`** (precision 0.67), namun memiliki tantangan dalam menghindari *False Positives* (585 berita palsu dikategorikan sebagai asli) dan *False Negatives* (904 berita asli dikategorikan sebagai palsu). Hal ini menyoroti bahwa meskipun model cukup baik dalam menegaskan berita yang jelas palsu, terdapat kesulitan signifikan dalam membedakan berita `REAL` dari `FAKE` secara akurat, yang menyebabkan banyak disinformasi lolos dan berita asli salah dikategorikan.

### Rekomendasi

1.  **Fokus pada Pengurangan *False Positives* (FP) Berita FAKE:**
    * **Tindakan:** Perkuat instruksi *prompt* model untuk lebih konservatif dalam mengkategorikan berita sebagai `REAL` jika ada elemen yang sedikit saja meragukan, atau tambahkan contoh *few-shot* yang lebih kompleks untuk skenario abu-abu.
    * **Dampak:** Mampu mengurangi jumlah berita palsu yang lolos sebagai berita nyata, sehingga secara langsung memerangi penyebaran disinformasi yang merugikan.

2.  **Kaji Ulang dan Perbaiki *False Negatives* (FN) Berita REAL:**
    * **Tindakan:** Lakukan analisis mendalam terhadap 904 berita `REAL` yang salah diklasifikasikan sebagai `FAKE` untuk mengidentifikasi pola umum (misalnya, gaya penulisan, topik, atau sumber yang kurang umum). Kemudian, sesuaikan *prompt* atau tambahkan contoh *few-shot* yang menargetkan pola-pola ini agar model lebih akurat dalam mengidentifikasi berita `REAL` yang sah.
    * **Dampak:** Meningkatkan keandalan model dalam mengidentifikasi informasi yang benar, membangun kepercayaan pengguna, dan mencegah berita asli disalahpahami sebagai hoaks.

## 🚀 Langkah Selanjutnya

* **Ekspansi Dataset:** Menggunakan dataset yang lebih besar dan lebih beragam untuk pelatihan dan evaluasi.
* **Fine-tuning Model:** Mengeksplorasi *fine-tuning* model Granite (jika memungkinkan dengan sumber daya yang tersedia) untuk kinerja yang lebih optimal.
* **Analisis Kesalahan Mendalam:** Melakukan analisis kualitatif lebih lanjut pada jenis kesalahan klasifikasi untuk mengungkap akar masalah.
* **Integrasi Real-time:** Mengembangkan mekanisme untuk mengintegrasikan model ke dalam sistem deteksi berita palsu real-time.
