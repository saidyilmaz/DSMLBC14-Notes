############################################
# User-Based Collaborative Filtering
#############################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
# Adım 6: Çalışmanın Fonksiyonlaştırılması

#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)

# Gerekli olan user-title matrixini aşağıdaki şekilde oluşturuyoruz (bkz. item_based_recommender_notes.py)
movie = pd.read_csv('Week 5 - Recommender Systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Week 5 - Recommender Systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId")
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 10000].index
common_movies = df[~df["title"].isin(rare_movies)]
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# rastgele bir kullanıcı seçelim
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)


#############################################
# Adım 2: Öneri Yapılacak Kullanıcının İzlediği Filmlerin Belirlenmesi
#############################################
random_user = 28941 # seçtiğimiz kullanıcının ID'si
user_movie_df  # kullanacağımız user-title matrixi
random_user_df = user_movie_df[user_movie_df.index == random_user]  # bu matrixteki ilgili kullanıcının film puanları

# İlgili kullanıcının izlediği filmleri görelim
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Örnek bir filme kaç puan verdiğine bakalım
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "Silence of the Lambs, The (1991)"]

# Toplamda kaç film izlediğini görelim
len(movies_watched)



#############################################
# Adım 3: Aynı Filmleri İzleyen Diğer Kullanıcıların Verisine ve Id'lerine Erişmek
#############################################

# En başta oluşturuduğumuz matrixten yalnızca ilgili kullanıcının izlediği filmlerin bilgisini alalım
movies_watched_df = user_movie_df[movies_watched]

# Burada ilgili kullanıcının izlediği filmlerden en az bir tanesini bile izlemiş olan bütün kullanıcılar mevcuttur.

# Bu listedeki her bir kullanıcının seçtiğimiz kullanıcıyla kaç tane ortak film izlediğine bakalım:
user_movie_count = movies_watched_df.T.notnull().sum()

# userId'leri bir değişken olarak DataFrame'e koyalım
user_movie_count = user_movie_count.reset_index()

# ortak film sayısı değişkeninin ismini movie_count olarak atayalım
user_movie_count.columns = ["userId", "movie_count"]

# seçtiğimiz kullanıcı ile en az 20 ortak film izlemiş olanları filtreleyelim
user_movie_count[user_movie_count["movie_count"] > 20].sort_values("movie_count", ascending=False)

# seçtiğimiz kullanıcının izlediği bütün filmleri izleyen kullanıcı sayısı
user_movie_count[user_movie_count["movie_count"] == 28].count()

# seçtiğimiz kullanıcı ile en az 20 ortak film izlemiş olanların ID'lerini toplayalım
users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

# ortak izlenen film sayısını seçili kullanıcnın toplam izlediği film sayısının yüzde 60'ı şeklinde de seçebiliriz:
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
# perc = len(movies_watched) * 60 / 100

#############################################
# Adım 4: Öneri Yapılacak Kullanıcı ile En Benzer Davranışlı Kullanıcıların Belirlenmesi
#############################################

# Bunun için 3 adım gerçekleştireceğiz:
# 1. Seçtiğimiz kullanıcı ve diğer kullanıcıların verilerini bir araya getireceğiz.
# 2. Korelasyon df'ini oluşturacağız.
# 3. En benzer bullanıcıları (Top Users) bulacağız

# Seçtiğimiz kullanınıcı ve diğer kullanıcıların verilerini bir araya getirdik:
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])

# Filmler üzerinden işlem yapacağımız için filmleri index'e, kullanıcıları sütuna alarak korelasyon matrix hesaplayalım
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()  # okunabilirliği düşük bir sonuç oldu

# Hesapladığımız değerlere "corr" ismini verdik
corr_df = pd.DataFrame(corr_df, columns=["corr"])

# İlk iki sütunda bütün kullanıcıların ID'leri vardır, bunları da aşağıdaki şekilde isimlendirelim
corr_df.index.names = ['user_id_1', 'user_id_2']

# Bu DataFrame'in indexlerini resetleyelim
corr_df = corr_df.reset_index()

# Şimdi bu verisetinden seçtiğimiz kullanıcının diğer kullanıcılarla olan korelasyonlarına
# bakıp 0.65'ten büyük olanları gözlemleyelim
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

# korelasyonları artacak şekilde sıralayalım (dikkat: en üstte seçtiğimiz kullanıcının kendisi var!)
top_users = top_users.sort_values(by='corr', ascending=False)

# değişken sütunun ismini tekrar "userId" yapıyoruz ki ilerleyen adımlarda başlangıçtaki veri setimizle kullanabilelim
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

# Burada seçtiğimiz kullanıcıyla aynı filmleri izleyen kişiler içerisinden
# verdikleri puanlara göre korelasyonları en yüksek olanlar vardır. Bunların yanına bir de filmlerin ID'lerini
# ve verilen puanları ekleyelim
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

# Seçtiğimiz kişinin kendisinin de bu listede olduğunu söylemiştik, bunu listeden çıkaralım
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]


#############################################
# Adım 5: Weighted Average Recommendation Score'un Hesaplanması
#############################################

# Aşağıdaki işlem sayesinde korelasyonun etkisi altında rating'lerin ne olduğunu hesaplamış olduk
top_users_ratings["weighted_rating"] = top_users_ratings["corr"] * top_users_ratings["rating"]

# Aynı filme birden fazla kullanıcı oy vermiş olabileceği için movieId üzerinden groupby alıp
# ağırlıklı puanların ortalamasını alabiliriz
recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

# 3.5'tan büyük olan skorlara bakalım ve bunları büyükten küçüğe sıralayalım
movies_to_be_recommended = recommendation_df[recommendation_df["weighted_rating"] > 3.5].sort_values("weighted_rating", ascending=False)

# Son olarak burada elde ettiğmiz movieId'lerin hangi filmlere denk geldiğini bulmak
movies_to_be_recommended.merge(movie[["movieId", "title"]])



#############################################
# Adım 6: Çalışmanın Fonksiyonlaştırılması
#############################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()

# perc = len(movies_watched) * 60 / 100
# users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]


def user_based_recommender(random_user, user_movie_df, ratio=60, cor_th=0.65, score=3.5):
    import pandas as pd
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    perc = len(movies_watched) * ratio / 100
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns=["corr"])
    corr_df.index.names = ['user_id_1', 'user_id_2']
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop=True)

    top_users = top_users.sort_values(by='corr', ascending=False)
    top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
    rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
    top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']

    recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
    recommendation_df = recommendation_df.reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score].sort_values("weighted_rating", ascending=False)
    movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
    return movies_to_be_recommend.merge(movie[["movieId", "title"]])



random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df, cor_th=0.70, score=4)


