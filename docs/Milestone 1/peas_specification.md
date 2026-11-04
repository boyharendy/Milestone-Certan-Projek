# PEAS Specification

## 1. Performance Measure (P)

1.	Total waktu evakuasi dari titik bahaya ke shelter (dalam menit).
2.	Persentase warga yang berhasil mencapai shelter sebelum estimasi waktu tsunami tiba.
3.	Minimisasi jarak tempuh rute (meter/km).
4.	Minimisasi skor risiko kumulatif sepanjang rute (akumulasi bobot risiko jalur yang dilalui).
5.	Tingkat distribusi beban jalur (menghindari satu jalur menampung terlalu banyak orang sekaligus).

## 2. Environment (E)

1.	Jaringan jalan/jalur evakuasi Kota Banda Aceh yang dimodelkan sebagai graf.
2.	Sistem peringatan dini BMKG (waktu dan estimasi kedatangan tsunami).
3.	Data kondisi infrastruktur jalur secara real-time (aman atau rusak).
4.	Data kepadatan pejalan kaki dan kendaraan per jalur.
5.	Lokasi serta kapasitas shelter atau titik kumpul evakuasi.

## 3. Actuators (A)

1.	Menampilkan rekomendasi rute evakuasi kepada warga melalui aplikasi atau notifikasi.
2.	Melakukan reroute otomatis saat kondisi jalur berubah, seperti rusak atau padat.
3.	Menandai dan memperbarui status sebuah jalur sebagai “tidak aman” dalam sistem.
4.	Mengirim prioritas alert ke petugas lapangan di titik jalur kritis.


## 4. Sensors (S)

1.	Lokasi awal atau titik bahaya warga (GPS).
2.	Status kondisi jalur secara real-time (dari laporan lapangan, sensor IoT, atau BPBD).
3.	Data kepadatan lalu lintas per jalur.
4.	Estimasi waktu kedatangan tsunami dari BMKG.
5.	Status okupansi atau kapasitas shelter saat ini.


## 5. Analisis Sifat Lingkungan Tugas

### Partially Observable (bukan Fully Observable)
Agen tidak memiliki akses penuh terhadap seluruh kondisi kota secara real-time. Status kerusakan jalur, kepadatan warga di titik tertentu, atau posisi pasti setiap individu tidak selalu tersedia secara instan dan akurat. Oleh karena itu, agen bergantung pada sensor dan laporan yang mungkin mengalami keterlambatan atau ketidaklengkapan data.

### Stochastic (bukan Deterministic)
Hasil dari suatu aksi tidak selalu pasti. Misalnya, rekomendasi rute yang diberikan mungkin tidak dipatuhi oleh warga, atau kondisi jalur dapat berubah secara tiba-tiba akibat longsor, kemacetan, atau gangguan lain di luar prediksi model.

### Sequential (bukan Episodic)
Keputusan rute yang direkomendasikan pada satu waktu akan memengaruhi kondisi berikutnya, seperti kepadatan jalur dan ketersediaan shelter. Dengan demikian, keputusan agen tidak berdiri sendiri, melainkan saling memengaruhi dalam rangkaian peristiwa yang berkelanjutan.

### Dynamic (bukan Static)
Lingkungan terus berubah selama agen bekerja. Kondisi jalur, kepadatan massa, dan waktu tersisa sebelum tsunami tiba bergerak secara dinamis, sehingga agen harus mempertimbangkan perubahan tersebut secara terus-menerus dan tidak boleh mengasumsikan bahwa dunia tetap diam.

### Discrete (bukan Continuous)
Ruang keadaan dimodelkan sebagai graf dengan node yang mewakili titik, persimpangan, atau shelter, serta edge yang mewakili segmen jalur. Aksi yang dipilih agen juga berasal dari himpunan pilihan yang terbatas dan bersifat diskrit, bukan ruang koordinat kontinu secara tak terbatas.

### Multi-agent
Terdapat banyak warga yang bergerak secara bersamaan dan saling memengaruhi. Kepadatan jalur dipengaruhi oleh keputusan kolektif banyak individu, sementara petugas lapangan BPBD juga mengambil tindakan dalam lingkungan yang sama. Oleh karena itu, tugas ini memiliki karakter multi-agent.

