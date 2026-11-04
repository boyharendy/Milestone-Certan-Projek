# Problem Framing
### *Sistem Rekomendasi Rute Evakuasi Tsunami Berbasis Agen Pencarian Cerdas: Studi Kasus Kota Banda Aceh*


## 1. Profil Domain Bisnis

BPBD Kota Banda Aceh merupakan instansi pemerintah daerah yang bertanggung jawab atas mitigasi dan penanggulangan bencana di wilayah pesisir Kota Banda Aceh. Kota ini memiliki risiko tinggi terhadap gempa dan tsunami, mengingat sejarah bencana tsunami tahun 2004 yang menimbulkan kerugian besar bagi masyarakat dan infrastruktur. Dalam upaya mitigasi, BPBD mengoperasikan sistem peringatan dini berbasis sirene yang tersebar di beberapa titik strategis kota, seperti kawasan Jalan Cut Mutia, Taman Putroe Phang, Kantor Camat Baiturrahman, Indrapuri, Keutapang, dan Kilometer Nol. Selain itu, instansi ini juga menyediakan peta jalur evakuasi tsunami yang dikelompokkan berdasarkan sektor wilayah kota, mulai dari Sektor A hingga E. Meskipun sistem tersebut telah tersedia, pendekatan yang digunakan masih bersifat statis dan belum sepenuhnya mampu memberikan rekomendasi rute yang adaptif terhadap kondisi real-time saat bencana terjadi.

## 2. Alur Proses Saat Ini (As-Is)

1.	Sensor gempa     dari BMKG atau lembaga terkait mendeteksi potensi tsunami, lalu BPBD Kota Banda Aceh menerima peringatan dini.
2.	Petugas BPBD mengaktifkan sirene peringatan dini di titik-titik strategis di kota.
3.	Warga merujuk pada peta evakuasi statis per sektor untuk menentukan arah evakuasi menuju shelter atau dataran tinggi terdekat.
4.	Peta evakuasi yang digunakan masih bersifat statis dan seragam untuk semua kondisi, sehingga belum dapat menyesuaikan rekomendasi rute secara real-time berdasarkan kondisi yang berubah, seperti jalur yang rusak, kepadatan massa, atau hambatan lalu lintas.
5.	Petugas lapangan harus berkoordinasi secara manual saat muncul kendala di lapangan, seperti kemacetan atau jalur yang terputus. Kegiatan simulasi dan uji sirene rutin dilakukan secara berkala, tetapi bersifat latihan dan belum menjadi sistem rekomendasi rute adaptif yang berjalan secara real-time.


## 3. Pain Points

1.	Rute evakuasi bersifat statis dan tidak responsif terhadap kondisi real-time peta jalur evakuasi per sektor hanya memberikan panduan umum, sehingga tidak dapat menyesuaikan rekomendasi saat situasi berubah.
2.	Belum ada personalisasi rute berdasarkan lokasi individu warga dalam satu sektor dapat memiliki kebutuhan rute yang berbeda tergantung kondisi di lapangan, namun peta yang digunakan sama untuk semua.
3.	Potensi penumpukan massa pada jalur yang populer banyak warga cenderung memilih jalur yang sama, sehingga terjadi bottleneck dan memperlambat proses evakuasi.
4.	Risiko jalur yang berubah tidak terdeteksi secara otomatis  jalur yang rusak akibat gempa, longsor, atau kerusakan infrastruktur tidak selalu terupdate dalam peta, sehingga dapat mengarahkan warga ke jalur yang berbahaya.
5.	Ketergantungan pada keputusan manual proses pengambilan keputusan dan koordinasi masih dilakukan secara manual, padahal waktu evakuasi tsunami sangat terbatas dan memerlukan respon yang cepat.


## 4. Justifikasi AI (Search-based Agent)

Proses evakuasi yang saat ini masih bergantung pada sirene dan peta statis tidak mampu memproses secara simultan tiga variabel kritis, yaitu jarak, risiko jalur, dan kepadatan massa, dalam hitungan detik. Oleh karena itu, diperlukan sebuah agen cerdas berbasis algoritma pencarian (search-based agent) yang dapat memodelkan jaringan jalur evakuasi sebagai graf berbobot dan menghitung rute paling aman dan paling cepat dari titik bahaya menuju shelter terdekat. Dengan pendekatan ini, rekomendasi dapat diperbarui secara dinamis sesuai perubahan kondisi jalur, sehingga proses evakuasi menjadi lebih efektif dan responsif dibandingkan pendekatan manual yang masih bersifat statis.