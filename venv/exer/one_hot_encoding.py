import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

data = pd.read_csv('../data/onehotenc_data.csv')
print(data)

preprocess = make_column_transformer(
    (StandardScaler(), ['CreditScore']),
    (OneHotEncoder(), ['Zone'])
)

data = preprocess.fit_transform(data)

