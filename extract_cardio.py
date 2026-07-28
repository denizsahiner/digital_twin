import pandas as pd

df = pd.read_csv('../datasets/cardio_train.csv', sep=';')
df = df[(df['ap_hi'] >= 70) & (df['ap_hi'] <= 220)]
df = df[(df['ap_lo'] >= 40) & (df['ap_lo'] <= 150)]
df = df[(df['height'] >= 140) & (df['height'] <= 200)]
df = df[(df['weight'] >= 40) & (df['weight'] <= 200)]

df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
df['map'] = df['ap_lo'] + (df['pulse_pressure'] / 3)
df['age_years'] = (df['age'] / 365.25).astype(int)
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
df['PP'] = df['ap_hi'] - df['ap_lo']


df.to_csv('../datasets/cardio_train_feature.csv', sep=';', index=False)