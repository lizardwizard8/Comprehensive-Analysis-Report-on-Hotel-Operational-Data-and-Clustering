# KMeans Clustering

# PyCaret'in kümeleme ortamını hazırlıyoruz. 
# normalize=True diyerek veriyi ölçeklendirmiş oluyoruz. 
# session_id vererek de aynı sonuçları almayı garanti ediyoruz.
exp_clustering = setup(data=df_processed, normalize=True, session_id=123)

# KMeans modelini oluşturuyoruz.
# Kaç küme istiyorsak num_clusters ile belirtiyoruz (7 küme seçtik).
kmeans_model = create_model('kmeans', num_clusters=7)

# Kümeleme sonucunu görselleştirmek için plot_model fonksiyonunu kullanıyoruz.
plot_model(kmeans_model, plot='cluster')

# Modeli uygulayıp her veriye küme etiketi atıyoruz.
df_clustered = assign_model(kmeans_model)

# Küme etiketleri eklenmiş veriyi Excel'e kaydediyoruz.
df_clustered.to_excel("C:\\Users\\gunal\\Desktop\\Bitirme Projesi\\hotel_clustered_results.xlsx", index=False)



# DBSCAN Clustering

from sklearn.neighbors import NearestNeighbors

# min_samples ile aynı olacak şekilde k seçiyoruz ( 10 dedik).
k = 10
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(df_processed)
distances, indices = neighbors_fit.kneighbors(df_processed)

# K-en yakın komşu mesafelerini sıralayıp çiziyoruz.
k_distances = np.sort(distances[:, k-1])
plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.xlabel('Veri Noktaları (sıralı)')
plt.ylabel(f'{k}. En Yakın Komşu Mesafesi')
plt.title('DBSCAN için K-Mesafe Grafiği')
plt.show()

# DBSCAN modelini belirlediğimiz eps ve min_samples ile oluşturuyoruz.
dbscan_model = create_model(
    'dbscan', 
    eps=10,         # 3 civarlarında deneyerek en uygun değeri bulabiliriz.
    min_samples=10  # Yukarıdaki k ile aynı olacak.
)

# DBSCAN modelini değerlendiriyoruz.
evaluate_model(dbscan_model)

# Kümeleme sonuçlarını görselleştiriyoruz.
plot_model(dbscan_model, plot='cluster')

# Küme etiketlerini veriye ekleyip ekrana yazdırıyoruz.
clustered_data = assign_model(dbscan_model)
print(clustered_data.head())


# Agglomerative Clustering

from pycaret.clustering import *

# PyCaret kümeleme ortamını hazırlıyoruz.
exp_clustering = setup(data=df_processed, normalize=True, session_id=123)

# Hiyerarşik kümeleme (Agglomerative Clustering) modelini oluşturuyoruz.
# Kaç küme olacağını num_clusters ile belirtiyoruz (burada 4 seçtik ama değiştirilebilir).
agglom_model = create_model('hclust', num_clusters=4)

# Modelin performansına bakmak için değerlendiriyoruz.
evaluate_model(agglom_model)

# Kümeleme sonucunu görselleştiriyoruz.
plot_model(agglom_model, plot='cluster')

# Küme etiketlerini veriye ekleyip ekrana yazdırıyoruz.
clustered_data = assign_model(agglom_model)
print(clustered_data.head())
