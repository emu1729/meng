import pandas as pd
import numpy as np
import datetime

df = pd.read_csv('./data/Incident-level/MIDIP_4.01.csv');

#Remove all unknown start dates
df = df.loc[df['StDay'] != -9]
df = df.reset_index(drop=True)

#Find list of all countries
countries = df['StAbb'].unique()

#Change table to just start day, month, and year
df_country_days = df[['StAbb', 'StDay', 'StMon', 'StYear']]

#Rename to day, month, year
df_country_days.rename(columns={'StAbb': 'state', 'StDay': 'day', 'StMon': 'month', 'StYear': 'year'}, inplace=True)

#countries
df_countries = df_country_days[['state']]

#time
df_days = df_country_days[['day', 'month', 'year']]

#change to datetime
df_dates = pd.to_datetime(df_days)
day_1 = df_dates[0]
days = []
for i in range(df_dates.shape[0]):
	days.append((df_dates[i] - day_1).days)
days = pd.Series(days)
df_countries['days'] = days.values

#construct time sequence
time_seq = []
red_time_seq = []
red_countries = []
for country in countries:
	df_temp = df_countries.loc[df_countries['state'] == country]
	times = df_temp['days'].tolist()
	times = sorted(set(times))
	time_seq = time_seq + [times]
	if len(times) >= 400:
		red_time_seq = red_time_seq + [times]
		red_countries.append(country)

print(red_countries)
print(red_time_seq)

#construct country sizes
mil = 10e6
pop = {'BEL': 11.35*mil, 'CAN': 36.29*mil, 'DEN': 5.7*mil, 'FRN': 66.9*mil, 'GMY': 82.67*mil, 'GRC': 10.75*mil,
	'ITA': 60.6*mil}
