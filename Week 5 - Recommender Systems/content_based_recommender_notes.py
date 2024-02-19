#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv("Week 5 - Recommender Systems/datasets/the_movies_dataset/movies_metadata.csv", low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape

df["overview"].head()

# Daha sonra kullanmak üzere bir tfidf metodu tanımlayalım:
tfidf = TfidfVectorizer(stop_words="english")  # burada ingilizce dilindeki tek başına anlam ifade etmeyecek
                                                # kelimeleri (and, or, in, on vb.) çıkarmak için stop_words argümanına
                                                # bu işlemi ingilizce diline göre yap komutu veriyoruz.
                                                # Bu oluşturulacak olan TF-IDF matrixinde büyük ölçüde azalma sağlar

# df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')  # overview kısmı boş olan filmlerin açıklamalarını space ile doldurduk

tfidf_matrix = tfidf.fit_transform(df['overview'])  # TF-IDF matrixi oluşturduk. Satırlar filmlerin açıklamalarını
                                                    # ve sütunlarda bütün açıklamalarda geçen kelimeleri barındırıyor.

tfidf_matrix.shape

df['title'].shape

tfidf.get_feature_names()  # sütunlarda yer alan kelimeleri gözlemlemek için

tfidf_matrix.toarray()  # matrixi bir array olarak tanımlamak için


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)
# Her bir film için elde ettiğimiz tfidf vektörüyle diğer bütün vektörlerin cosine similarty değerini hesaplayıp
# bir matrixe kaydettik. n film sayısı ise bu matrix nxn boyutunda olmalıdır.

cosine_sim.shape
cosine_sim[1]  # ikinci satırdaki filmlerin diğer filmlerle olan cosine similarty değerlerini görüyoruz
                # bu vektörün ikinci componentındaki değer bu filmin kendisiyle olan similarty değerini
                # yani 1 değerini gösterir


#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

# Yukarıda bulduğumuz cosine similarty matrixte filmlerin isimleri yok yalnızca açıklamaları var. Şimdi bu sorunu
# gidermek için aşağıdaki işlemleri yapacağız:

indices = pd.Series(df.index, index=df['title'])  # filmlerin isimlerini bir PandasSeries içerisinde koyalım

indices.index.value_counts()  # burada görüyoruz ki bazı filmlerin isimleri birden fazla kez listelenmiş.
                                # Bunun sebebi aynı film birden fazla kez çekilmiş ya da farklı filmler aynı şekilde
                                # isimlendirilmiş olabilir

indices = indices[~indices.index.duplicated(keep='last')]  # tekrar edenlerden sadece en güncel olanını tutmayı tercih edelim

indices["Cinderella"]  # Artık Cindirella ismiyle arama yaptığımızda elimizde yalnızca sonunucusunun indexi var

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]  # Sherlock Holmes filminin similarty vektörünü elde etmiş olduk.
                            # Bunun okunabilirliği çok düşük olduğu için buradaki bilgileri bir DataFrame'e aktaralım:

similarity_scores = pd.DataFrame(cosine_sim[movie_index],
                                 columns=["score"])  # Burada Sherlock Holmes filminin diğer bütün filmlerle
                                                        # olan benzerlik skorları vardır

# şimdi bu filmle similarty skoru en yüksek olan ilk on filmin indexlerine erişelim
movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index

# son olarak bu indexlere karşılık gelecek film isimlerini gözlemleyelim
df['title'].iloc[movie_indices]

#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)
# 1 [90, 12, 23, 45, 67]
# 2 [90, 12, 23, 45, 67]
# 3
