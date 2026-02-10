# BiLSTM-Sentiment-Analysis-X
Analisis Sentimen Publik terhadap ChatGPT berdasarkan Tweet pada Platform Media Sosial 𝕏 Menggunakan Model Bidirectional Long Short-Term Memory (BiLSTM) 



**1. Pendahuluan**  
Pesatnya perkembangan teknologi menghadirkan faktor-faktor baru dalam kehidupan bermasyarakat. Kecanggihan teknologi juga tampak jelas, tidak terkecuali pada bidang Artificial Intelligence (AI). Model ChatGPT tentunya menjadi bukti nyata betapa majunya teknologi, khususnya dalam bidang deep learning. Popularitas ChatGPT menjadi bukti nyata peran teknologi dalam hidup manusia. Namun, Integritas serta etika semakin dipertanyakan dalam hal penggunaan ChatGPT ditengah betapa bergunannya model ini. Kebijakan pada negara maupaun OpenAI bergantung juga pada pandangan publik terkait ChatGPT itu sendiri. 

**2. Variabel**  
Dataset from Huggingface: https://huggingface.co/datasets/prithivMLmods/GPT-Sentiment-Analysis-200K  

Dataset sudah diberikan label. Jumlah data ada sebanyak 219,294 dan terdapat 3 kolom yaitu,
- id: Id atau nilai unik tiap tweet
- Tweet: Konten tweet berupa teks 
- Labels: Kategori sentimen tweet (Neutral, Good, Bad)

**3. Metode**  
Berikut proses pengerjaannya:
<img width="810" height="374" alt="image" src="https://github.com/user-attachments/assets/dfb44c8e-7855-4da3-9033-d39da72b7bc3" />

*Text processing* merupakan langkah krusial dalam melakukan analisa data teks. Pada projek ini, tahapan berikut dilakukan pada dataset:
1) *Lowercasing* atau pembuatan kata pada teks menjadi huruf kecil semua.
2) Penghapusan spasi ekstra atau lebih
3) Penghapusan URL atau Hyperlink
4) Penghapusan HTML Markup
5) Penghapusan Emoji
6) Memperpanjang kata (e.g. isn’t menjadi is not)
7) Buang karaketer yang bukan bagian dari alfabet
8) Penghapusan hashtags dan mention
9) Penghapusan karaketer berupa digit angka

Kemudian dilakukan penghapusan *stop words*. *Stop words* merupakan kata-kata yang sering muncul, tetapi eksisten demi keperluan grammar penulisan saja dan tidak memberikan informasi yang berguna untuk pemodelan. Contoh *stop words* pada bahasa inggris seperti kata, *the*, *a*, *would*, *is*, *he*, *she*, *it*. Analisis data menggunakan Kumpulan stop words yang disediakan oleh package nltk. Kemudian, proses lemmatisasi kata dilakukan. Lemmatisasi merupakan proses pengubahan kata jamak atau kata dengan imbuhan menjadi ke bentuk dasarnya (*e.g.* *food’s* menjadi *food*, *Improvisation* menjadi *Improve*). Proses ini membantu dalam pelatihan model sebab hanya menggunakan kata dasarnya saja sebagai konteksnya dan tidak memperhatikan imbuhan.  

Data teks telah bersih setelah dilakukan lemmatisasi dan penghapusan noise sehingga proses tokenisasi dilakukan selanjutnya. Tahap tokenisasi bertujuan untuk mengubah kalimat menjadi pecahan-pecahan kata yang berdiri secara individual. Pecahan tersebut disebut token. Proses ini krusial sebab menyederhanakan data yang akan diproses oleh model nantinya. Namun, pelatihan model deep learning memerlukan dimensi yang sama untuk setiap data latihannya. Penelitian membatasi jumlah kata yang ditoken sebanyak 4000 sebab panjang kalimat dari setiap tweet tentunya berbeda-beda. Selanjutnya, padding dan truncating diterapkan untuk membuat beberapa kata saja yang digunakan untuk pelatihan. Namun, pada pelatihan model untuk penelitian ini, sekuens terpanjang dari seluruh data digunakan sebagai batas maksimum sebab ingin digunakan semua kata untuk pemodelan. Oleh sebab itu, maksimum panjang sekuensnya adalah 35

**4. Hasil**  
Terdapat 3 plot yang masing-masing menunjukkan kata terbanyak dari tiap tweet yang berlabel Neutral, Good, dan Bad. Pada gambar, judul barplot menggunakan ‘Negatif’ sebagai sentimen untuk label ‘Bad’ dan ‘Positif untuk label ‘Good’. Visualisasi tidak mengikutsertakan kata 'chatgpt', 'gpt', 'ai', 'google', 'opena', 'openai', 'chatbot', 'via' sebab mendominasi pada data teks sehingga tidak informatif.  

<img width="1990" height="490" alt="image" src="https://github.com/user-attachments/assets/28c61c06-fa19-48d0-9af1-a911e507537c" />  

Hasil visualisasi barplot menunjukkan bahwa, sentimen publik positif terhadap ChatGPT banyak menggunakan kata ‘like’, ‘good’, ‘great’, dan ‘new’. Hal ini mengindikasikan bahwa tampaknya performa ChatGPT sudah baik dan bagus serta adanya fitur baru memberikan kesan baik untuk public. Akan tetapi, hasil barplot untuk sentimen negatif tampak sentimen publik negatif terhadap ChatGPT berkaitan dengan penggunaan ChatGPT, jawaban yang diberikan oleh ChatGPT, saat bertanya kepada ChatGPT, dan lainnnya.  

<img width="944" height="505" alt="image" src="https://github.com/user-attachments/assets/390a73b8-eca9-4613-a702-eec659c698c4" />  

Word Cloud juga dibuat untuk melihat kata-kata terumum secara keseluruhan dari tweet terkait ChatGPT dengan tidak mengikutsertakan kata-kata sama seperti dalam pembuatan barplot. Hasil visualisasi dapat dilihat pada gambar di atas. Word Cloud menunjukkan banyak menggunakan kata seperti ‘like’, ‘new’, ‘asked’, ‘write’ yang mana sesuai dengan konteks data. ChatGPT pasti berkaitan dengan fitur terbarunya serta menanyakan pertanyaan dengan menulis prompt.  

Gambar berikut menunjukkan dengan jelas adanya *imbalanced* pada data. 
<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/233367bc-8e36-4411-87a8-3ea0e04a6a47" />  
Tweet dengan label ‘Bad’ ada sekitar 49.2%, ‘Neutral’ ada sekitar 25.3%, dan label ‘Good’ ada sekitar 25.5%. Oleh sebab itu,  metode *Synthetic Minority Over-sampling* (SMOTE) digunakan untuk menyeimbangkan distribusi labelnya.

<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/acb847b7-b574-4d7d-b808-5710ed8ff916" />

Selanjutnya, model BiLSTM dikonstruksi. Model dibuat dengan pertama memberikan lapisan Embedding dengan jumlah kosakata 4000 dan panjang vektor masing-masing kata adalah 35. Lapisan Bidirectional LSTM ditambahkan pada model dengan jumlah neuron sebanyak 256 sebab proses dilakukan backward dan forward secara paralel. Pada lalpisan ini ditentukan 50% input data dibuang atau tidak digunakan pada saat training. Hal ini bertujuan untuk regularisasi pelatihan model dan menghindari overfitting. Setelah itu, lapisan SelfAttention ditambahkan agar memahami konteks teks lebih baik dengan cara melihat kalimat secara keseluruhan. Lapisan ini juga memberikan bobot kepada token yang signifikan untuk memahami konteks teks. Kemudian, lapisan pooling ditambahkan agar dimensi menjadi lebih kecil dengan mengubah sekuens menjadi sebuah vektor. Ukuran setelah direduksi akan berdimensi 256 x 1 dan diambil fitur dengan nilai maksimum dari tiap sekuens.  

Batch Normalitzation dilakukan untuk stabilisasi training data. Selanjutnya, lapisan fully connected dengan 64 neuron ditambahkan dan dengan fungsi aktivasi Rectified Linear Unit (reLU) digunakan agar mendeteksi adanya pola non-linear atau pola-pola kompleks lainnya pada data. Lapisan Output berfungsi untuk mengeluarkan hasil final pada klasifikasi. Lapisan ini pada model memiliki tiga neuron dan fungsi aktivasi softmax dipilih sebab sentimen atau label memiliki lebih dari dua kategori. Model selanjutnya disatukan dan metode optimizer Adam dengan learning 0.001 dipilih untuk pelatihan model. Loss dihitung dengan ‘categorical_crossentropy’.  

<img width="643" height="242" alt="image" src="https://github.com/user-attachments/assets/a6c18a82-1d57-49f6-b02b-9c8783201c44" />  

Evaluasi model pertama adalah melihat akurasi serta loss saat pelatihan. Hasil akurasi menunjukkan ada tampak permasalahan overfitting pada validasi sebab berbeda cukup besar. Hal ini mungkin disebabkan kurangnya regulasi pada arsitektur model. Hasil ini juga mungkin disebabkan oleh kurang lengkap atau menyeluruh dalam proses persiapan data teks sehingga masih banyak kata-kata yang tidak informatif pada saat training. Namun, pergerakan akurasi dan loss terlihat semakin menuju pada titik yang sama.  

Rata-rata akurasi model mencapai 0.85 atau 85% yang mana sesuai dengan akurasi validasi saat pelatihan model. Model berarti berhasil memprediksi sebesar 85% dari total atau keseluruhan tweet. Hasil macro average di sekitar 0.84 dan weighted average 0.85 di semua kategori menunjukkan. Namun, Precision, Recall, dan F1-Score untuk label ‘Neutral’ masih kurang.

<img width="567" height="455" alt="image" src="https://github.com/user-attachments/assets/d6c1f363-f763-4641-9617-cf8f2e9271fb" />  








