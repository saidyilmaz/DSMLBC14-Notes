import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules

### Görev 1: Veriyi Hazırlama ###

# Adım 1 #
df_ = pd.read_excel("Week 5 - Recommender Systems/Bonus Project/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df = df[df["Country"] == "Germany"]

# Adım 2 #
df = df[~(df["StockCode"] == "POST")]

# Adım 3 #
df.isnull().sum()  # boş değer yok

# Adım 4 #
df = df[~(df["Invoice"].str.contains("C", na=False))]

# Adım 5 #
df = df[df["Price"] > 0]

# Adım 6 #
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

### Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme ###

# Adım 1 #
invoice_product_df = df.pivot_table(index="Invoice", columns="StockCode", values="Quantity", aggfunc=sum, fill_value=0)
invoice_product_df = invoice_product_df.applymap(lambda x: 1 if x > 0 else 0)

# Adım 2 #
apriori_df = apriori(invoice_product_df, min_support=0.01, use_colnames=True, low_memory=True)
rules_df = association_rules(apriori_df, metric="support", min_threshold=0.01)

### Görev 3: Ürün Önerme ###
def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])
    return recommendation_list[0:rec_count]

arl_recommender(rules_df, product_id=21987, rec_count=3)

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)

check_id(df, 21086)
