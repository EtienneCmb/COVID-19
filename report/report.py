"""Python report for monitoring COVID-19."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import json

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import seaborn as sns

# plt.style.use('seaborn-paper')
sns.set_style("whitegrid")
plt.rc('font', family="serif")


###############################################################################
# plotted elements
case = 'Confirmed'  # {'Confirmed', 'Deaths', 'Recovered'}
to_plt = 'worst'  # {'world', 'eu', 'na', 'asia', 'melt', 'worst'}
# plotted continents / countries
plt_continents = [
    'Europe', 'Asia', 'North America', 'South America', 'Africa', 'Oceania']
plt_eu = [
    'France', 'Italy', 'Spain', 'United Kingdom', 'Germany', 'Belgium',
    'Netherlands', 'Greece', 'Norway', 'Denmark', 'Sweden', 'Switzerland']
plt_na = ['US', 'Canada', 'Mexico']
plt_asia = ['China', 'India', 'Russia', 'Korea, South', 'Thailand', 'Turkey']
plt_melt = [
    'China', 'Korea, South', 'Italy', 'Spain', 'Germany', 'US', 'France',
    'United Kingdom', 'Canada', 'Australia']
# plotting settings
log_scale = True
day_freq = 2  # x-label day frequency
n_top = 10  # number of top / worst
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
file = f'time_series_19-covid-{case}.csv'
df = pd.read_csv(f"../csse_covid_19_data/csse_covid_19_time_series/{file}")
df.rename(columns={'Country/Region': 'Country'}, inplace=True)
# remove lat / long columns
df.drop(labels=['Lat', 'Long', 'Province/State'], axis=1, inplace=True)
# dates conversion
dates = pd.to_datetime(df.iloc[:, 1::].columns).strftime('%d/%m')
dates_repl = {k: i for k, i in zip(df.columns[1::], dates)}
df.rename(columns=dates_repl, inplace=True)
# insert continent column
cont = list(df['Country'].replace(repl, regex=True))
df.insert(loc=0, column='Continents', value=cont)


# build the subdf
if to_plt == 'world':
    sub_df = df.groupby('Continents').sum().loc[plt_continents].reset_index()
    id_var = 'Continents'
elif to_plt == 'eu':
    _df = df.set_index('Continents').loc['Europe']
    sub_df = _df.groupby('Country').sum().loc[plt_eu].reset_index()
    id_var = 'Country'
elif to_plt == 'na':
    _df = df.set_index('Continents').loc['North America']
    sub_df = _df.groupby('Country').sum().loc[plt_na].reset_index()
    id_var = 'Country'
elif to_plt == 'asia':
    _df = df.set_index('Continents').loc['Asia']
    sub_df = _df.groupby('Country').sum().loc[plt_asia].reset_index()
    id_var = 'Country'
elif to_plt == 'melt':
    _df = df.set_index('Country').loc[plt_melt]
    sub_df = _df.groupby('Country').sum().loc[plt_melt].reset_index()
    id_var = 'Country'
elif to_plt in ['top', 'worst']:
    _df = df.groupby('Country').sum().reset_index()
    _df.index = _df.iloc[:, -1] - _df.iloc[:, -2]
    if to_plt == 'worst':
        sub_df = _df.sort_index(ascending=False).iloc[0:n_top, :]
        sub_df = sub_df.reset_index(drop=True)
    id_var = 'Country'


# build the mlet dataframe, dates and color palette
sub_df = sub_df.melt(id_vars=id_var, var_name='Date', value_name='Count')
date_range = np.arange(len(dates))
palette = sns.color_palette("tab20", n_colors=len(sub_df))
# plot results
fig = plt.figure(figsize=(11, 9))
ax = plt.gca()
sns.factorplot(x="Date", y="Count", hue=id_var, data=sub_df, palette=palette,
               ax=ax)
plt.close(fig=2)
ax = plt.gca()
ax.set_xticks(date_range[::-1][::day_freq][::-1])
ax.set_xticklabels(dates[::-1][::day_freq][::-1], rotation=45, ha='right')
plt.xlabel(''), plt.ylabel(f'# {case.lower()}', fontsize=13)
plt.title('Monitoring COVID-19 since 22 Jan 2020', fontsize=15,
          fontweight='bold')
if log_scale:
    ax.set_yscale('log', basey=10)
    plt.ylim(.9)
    ax.yaxis.set_major_formatter(ScalarFormatter())

plt.show()
