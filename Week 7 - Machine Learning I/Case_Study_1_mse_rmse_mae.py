import pandas as pd
import math

df = pd.read_csv("Week 7-8-9 - Machine Learning/machine_learning/datasets/advertising.csv")
df.head()

new_df = df[["newspaper", "sales"]]

b = 25
w = 0.2

new_df["sales_pred"] = b + df["newspaper"] * w

new_df["error"] = new_df["sales_pred"] - new_df["sales"]

new_df["error^2"] = new_df["error"] ** 2

mse = (new_df["error^2"].mean())

rmse = math.sqrt(mse)

new_df["abs_error"] = abs(new_df["error"])

mae = new_df["abs_error"].mean()
