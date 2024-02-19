###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
pd.set_option("display.width", 500)

# filmlerin isimlerinin ve id'lerinin olduğu veri seti
movie = pd.read_csv('Week 5 - Recommender Systems/datasets/movie_lens_dataset/movie.csv')

# filmlere verilen puanların ve puanı veren kullanıcının id'sini barındıran veri seti
rating = pd.read_csv('Week 5 - Recommender Systems/datasets/movie_lens_dataset/rating.csv')

# bu iki dataframe'i birleştirip bu df üzerinde çalışacağız
df = movie.merge(rating, how="left", on="movieId")
df.head()

df["movieId"].nunique()  # veri setinde 27278 tane farklı film varmış
df["userId"].nunique()  # veri setinde 138493 tane farklı kullanıcı varmış


######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

# yukarıdaki df'te sadece bir kaç oy almış filmelerin de bilgisi bulunmaktadır.
# fakat bu filmler birazdan hesaplayacağımız matrixe yüzlerce boş satır/sütun gelmesine sebep olacaktır
# bu da veriyi işleme verimliliğimizi büyük ölçüde düşüreceğinden bu filmlerden kurtulmamız gerekir

df.head()
df.shape  # bu veri setinde yaklaşık 20 milyon yorum vardır

df["title"].nunique()  # film sayısı

df["title"].value_counts().head()  # her bir filme verilen oy sayısı

comment_counts = pd.DataFrame(df["title"].value_counts())  # her filme verilen oy sayısını bir DataFrame'e çevirdik

rare_movies = comment_counts[comment_counts["title"] <= 10000].index  # 1000'den az oy alan filmleri belirledik

common_movies = df[~df["title"].isin(rare_movies)]  # 1000'den fazla oy alan filmleri belirledik

common_movies.shape
common_movies["title"].nunique()  # 1000'den fazla oy alan 3159 film varmış
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns


######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"  # bir film seçtik
movie_name = user_movie_df[movie_name]  # seçtiğimiz filme verilen oyların bulunduğu sütunu aldık

# bu listenin diğer filmlerle olan korelasyonlarını hesaplatıp en yüksek 10 filmi listeleyelim
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# rastgele bir film seçerek aynı işlemi yapabiliriz
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

#
def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Matrix", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Week 5 - Recommender Systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Week 5 - Recommender Systems/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 10000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





