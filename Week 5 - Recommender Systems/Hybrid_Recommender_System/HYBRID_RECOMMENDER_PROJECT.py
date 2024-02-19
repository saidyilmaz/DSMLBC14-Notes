
#############################################
# PROJE: Hybrid Recommender System
#############################################

# ID'si verilen kullanıcı için item-based ve user-based recomender yöntemlerini kullanarak tahmin yapınız.
# 5 öneri user-based modelden 5 öneri de item-based modelden ele alınız ve nihai olarak 10 öneriyi 2 modelden yapınız.

#############################################
# Görev 1: Verinin Hazırlanması
#############################################

# Gerekli kütüphaneleri import edelim
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Adım 1: Movie ve Rating veri setlerini okutunuz.
# movieId, film adı ve filmin tür bilgilerini içeren veri seti
movie = pd.read_csv("Week 5 - Recommender Systems/Hybrid_Recommender_System/datasets/movie.csv")

# UserID, film adı, filme verilen oy ve zaman bilgisini içeren veri seti
rating = pd.read_csv("Week 5 - Recommender Systems/Hybrid_Recommender_System/datasets/rating.csv")



# Adım 2: Rating veri setine filmlerin isimlerini ve türünü movie film setini kullanrak ekleyiniz.
# Ratingdeki kullanıcıların oy kullandıkları filmlerin sadece id'si var.
# Idlere ait film isimlerini ve türünü movie veri setinden ekliyoruz.
df = movie.merge(rating, how="left", on="movieId")



# Adım 3: Herbir film için toplam kaç kişinin oy kullandığını hesaplayınız. Toplam oy kullanılma sayısı 10000'un altında
# olan filmleri veri setinden çıkarınız. Herbir film için toplam kaç kişinin oy kullanıldığını hesaplıyoruz.
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] < 10000].index


# Toplam oy kullanılma sayısı 10000'in altında olan filmlerin isimlerini rare_movies de tutuyoruz.
# Ve veri setinden çıkartıyoruz
common_movies = df[~df["title"].isin(rare_movies)]


# Adım 4: # index'te userID'lerin sutunlarda film isimlerinin ve değer olarakta ratinglerin bulunduğu
# dataframe için pivot table oluşturunuz.
user_movie_df = common_movies.pivot_table(values="rating", index="userId", columns="title")


# Adım 5: Yukarıda yapılan tüm işlemleri fonksiyonlaştıralım
def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv("Week 5 - Recommender Systems/Hybrid_Recommender_System/datasets/movie.csv")
    rating = pd.read_csv("Week 5 - Recommender Systems/Hybrid_Recommender_System/datasets/rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] < 10000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(values="rating", index="userId", columns="title")
    return user_movie_df


#############################################
# Görev 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################

# Adım 1: Rastgele bir kullanıcı id'si seçiniz.
random_user = 28941

# Adım 2: Seçilen kullanıcıya ait gözlem birimlerinden oluşan random_user_df adında yeni bir dataframe oluşturunuz.
random_user_df = user_movie_df[user_movie_df.index == random_user]

# Adım 3: Seçilen kullanıcının oy kullandığı filmleri movies_watched adında bir listeye atayınız.
movies_watched = list(random_user_df.columns[random_user_df.notna().any()])
len(movies_watched)

#############################################
# Görev 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# Adım 1: Seçilen kullanıcının izlediği fimlere ait sutunları user_movie_df'ten seçiniz ve movies_watched_df adında yeni bir dataframe oluşturuyoruz.
movies_watched_df = user_movie_df[movies_watched]

# Adım 2: Herbir kullancının seçili user'in izlediği filmlerin kaçını izlediği bilgisini taşıyan user_movie_count adında yeni bir dataframe oluşturunuz.
user_movie_count = movies_watched_df.T.notna().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# Adım 3: Seçilen kullanıcının oy verdiği filmlerin yüzde 60 ve üstünü izleyenleri benzer kullanıcılar olarak görüyoruz.
# Bu kullanıcıların id’lerinden user_same_movies adında bir liste oluşturunuz.
percentage = 60
user_same_movies = list(user_movie_count[user_movie_count["movie_count"] > (len(movies_watched) * percentage / 100)].index)
len(user_same_movies)



#############################################
# Görev 4: Öneri Yapılacak Kullanıcı ile En Benzer Kullanıcıların Belirlenmesi
#############################################

# Adım 1: user_same_movies listesi içerisindeki seçili user ile benzerlik gösteren kullanıcıların id’lerinin bulunacağı şekilde movies_watched_df dataframe’ini filtreleyiniz.
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(user_same_movies)], random_user_df[movies_watched]])

# Adım 2: Kullanıcıların birbirleri ile olan korelasyonlarının bulunacağı yeni bir corr_df dataframe’i oluşturunuz.
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.index.names = ["userId_1", "userId_2"]
corr_df = corr_df.reset_index()

# Adım 3: Seçili kullanıcı ile yüksek korelasyona sahip (0.65’in üzerinde olan) kullanıcıları filtreleyerek top_users adında yeni bir dataframe oluşturunuz.
corr_th = 0.65
top_users = corr_df[(corr_df["userId_1"] == random_user) & (corr_df["corr"] > corr_th)][["userId_2", "corr"]].sort_values("corr", ascending=False)

# Adım 4:  top_users dataframe’ine rating veri seti ile merge ediniz
top_users.columns = ["userId", "corr"]
top_users_ratings = top_users.merge(rating, how="left", on="userId")



#############################################
# Görev 5: Weighted Average Recommendation Score'un Hesaplanması ve İlk 5 Filmin Tutulması
#############################################

# Adım 1: Her bir kullanıcının corr ve rating değerlerinin çarpımından oluşan weighted_rating adında yeni bir değişken oluşturunuz.
top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]


# Adım 2: Film id’si ve her bir filme ait tüm kullanıcıların weighted rating’lerinin ortalama değerini içeren recommendation_df adında yeni bir
# dataframe oluşturunuz.
recommendation_df = top_users_ratings[["movieId", "weighted_rating"]]


# Adım 3: Adım3: recommendation_df içerisinde weighted rating'i 3.5'ten büyük olan filmleri seçiniz ve weighted rating’e göre sıralayınız.
# İlk 5 gözlemi movies_to_be_recommend olarak kaydediniz.
rating_th = 3.5
movies_to_be_recommended = recommendation_df[recommendation_df["weighted_rating"] > rating_th]. \
    sort_values("weighted_rating", ascending=False).head()

# Adım 4:  Tavsiye edilen 5 filmin isimlerini getiriniz.
movies_to_be_recommended.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Item-Based Recommendation
#############################################

# Kullanıcının en son izlediği ve en yüksek puan verdiği filmin adına göre item-based öneri yapınız.
user = 108170

# Adım 1: movie,rating veri setlerini okutunuz.
movie = pd.read_csv("Week 5 - Recommender Systems/Hybrid_Recommender_System/datasets/movie.csv")
rating = pd.read_csv("Week 5 - Recommender Systems/Hybrid_Recommender_System/datasets/rating.csv")

# Adım 2: Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınız.
top_rated_movies = df[(df["userId"] == user) & (df["rating"] == 5)]
last_top_rated_movie_id = top_rated_movies[top_rated_movies["timestamp"] == top_rated_movies["timestamp"].max()]["movieId"].iloc[0]
last_top_rated_movie_name = top_rated_movies[top_rated_movies["timestamp"] == top_rated_movies["timestamp"].max()]["title"]

# Adım 3 :User based recommendation bölümünde oluşturulan user_movie_df dataframe’ini seçilen film id’sine göre filtreleyiniz.
user_movie_df = common_movies.pivot_table(values="rating", index="userId", columns="movieId")
movie_df = user_movie_df.loc[last_top_rated_movie_id]

# Adım 4: Filtrelenen dataframe’i kullanarak seçili filmle diğer filmlerin korelasyonunu bulunuz ve sıralayınız.
recommended_movies = user_movie_df.corrwith(movie_df).sort_values(ascending=False)

# Adım 5: Seçili film’in kendisi haricinde ilk 5 film’I öneri olarak veriniz.
recommended_movies.head()
