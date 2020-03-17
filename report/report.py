"""Python report for monitoring COVID-19."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import json

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# plt.style.use('seaborn-paper')
sns.set_style("whitegrid")
plt.rc('font', family="serif")


###############################################################################
plt_continents = [
    'Asia', 'Europe', 'Oceania', 'Africa', 'North America', 'South America'
    ]
day_freq = 2
###############################################################################


###############################################################################
# load the needed data
###############################################################################
# Load the continents
with open("continents.json") as f:
    continents = json.load(f)
# replacing patterns in order to infer continents
repl = {}
for cont, couns in continents.items():
    for coun in couns:
        repl[coun] = cont

# load the covid time-series
df = pd.read_csv("../csse_covid_19_data/csse_covid_19_time_series/time_series_19-covid-Confirmed.csv")
# split into countries / dates
df_names = df.iloc[:, :2]
df_names['Continents'] = df['Country/Region'].replace(repl, regex=True)
df_dates = df.iloc[:, 4::]
df_dates.index = df_names['Continents']
# convert columns into proper date times
dates = pd.to_datetime(df_dates.columns).strftime('%d/%m')
n_dates = len(dates)
date_range = np.arange(n_dates)
df_dates.columns = dates

gp_cont = df_dates.groupby(level=0).sum().loc[plt_continents, :]
arr_cont = np.array(gp_cont).T.astype(float)
arr_cont[arr_cont == 0] = np.nan

print(arr_cont[:, 0])

dif = np.gradient(arr_cont, np.arange(n_dates), axis=0)
plt.plot(dif[:, 0])
# arr = np.polyval(np.polyfit(date_range, arr_cont[:, 0], 1), date_range)
# print(arr.shape)
# plt.plot(arr)
plt.show()
exit()


gp_cont_cum = pd.DataFrame(arr_cont, index=dates,
                           columns=gp_cont.index).reset_index()
gp_cont_cum = gp_cont_cum.melt('index', var_name='Continents',
                               value_name='vals')

g = sns.factorplot(x="index", y="vals", hue='Continents', data=gp_cont_cum)
ax = plt.gca()
ax.set_yscale('log', basey=10)
ax.set_xticks(date_range[::-1][::day_freq][::-1])
ax.set_xticklabels(dates[::-1][::day_freq][::-1], rotation=-45)
plt.ylim(.5)
plt.ylabel("# confirmed cases", fontsize=13), plt.xlabel('')
plt.title('Monitoring COVID-19 since 22 Jan 2020', fontsize=15,
          fontweight='bold')

plt.show()

# print(df.groupby('Country/Region').sum())
