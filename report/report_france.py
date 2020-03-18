"""COVID-19 report for france.

COVID-19 data for France :
https://www.data.gouv.fr/en/datasets/cas-confirmes-dinfection-au-covid-19-par-region/
"""
import numpy as np
import pandas as pd
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import pygal
from pygal.style import BlueStyle, Style

sns.set_style("whitegrid")
plt.rc('font', family="serif")


###############################################################################
vlines = {'15/03': 'Elections', '16/03': 'Confinement'}
vlines_kw = dict(color='k', lw=1.)
###############################################################################


###############################################################################
# load the needed data
###############################################################################
# load the table containg departement number
df_table = pd.read_csv('../csse_covid_19_data/covid_france/depart_fr.csv')
df_table = df_table[['NUMÉRO', 'REGION']]
u_reg = np.unique(df_table['REGION']).tolist()
# convert depart num
depart_num = [str(k) if len(k) > 1 else f"0{k}" for k in df_table['NUMÉRO']]
df_table['NUMÉRO'] = depart_num
df_gp = df_table.groupby('REGION').groups
df_num_gp = {k: list(df_table['NUMÉRO'][i]) for k, i in df_gp.items()}
# load the latest covid-19 in france
df = pd.read_csv('../csse_covid_19_data/covid_france/covid19.csv')
df = df.loc[:, ['Date'] + u_reg]
# date conversion and add ranking
df['Date'] = pd.to_datetime(list(df['Date'])).strftime('%d/%m')
late_covid = df.iloc[-1, :]
late_covid_rank = df.iloc[-1, 1::].sort_values(ascending=False).argsort()
cols_r = {k: f"{k} ({n_k + 1})" for n_k, k in enumerate(late_covid_rank.index)}
df.rename(columns=cols_r, inplace=True)
# plotting conversion
df_reg = df.melt('Date', var_name='Régions', value_name='vals')
depart = np.unique(df_reg['Régions'])
n_depart = len(depart)
xlabs = list(df_reg['Date'])
palette = sns.color_palette("tab20")

arr = np.array(df.iloc[:, 1:])
# print(arr.shape)
# grow = arr[1::, :] / arr[0:-1, :]
# grow[~np.isfinite(grow)] = 1

# grow = np.diff(arr, axis=0)
# x = np.arange(0, grow.shape[0])
# x_new = np.linspace(0, grow.shape[0] - 1, num=20)
# print(x, x_new)
# f = interpolate.interp1d(x, grow, axis=0, kind='quadratic')
# grow = f(x_new)

# plt.plot(grow)
# ax = plt.gca()
# # ax.set_yscale('log', basey=10)
# plt.show()
# exit()



###############################################################################
# Plotting time-series
###############################################################################

fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(1, 2, left=0.05, bottom=0.05, right=0.99, top=0.90,
                  wspace=.01)
ax1 = plt.subplot(gs[0, 0])
g = sns.factorplot(x="Date", y="vals", hue='Régions', data=df_reg,
                   palette=palette, ax=ax1)
# ax1.set_yscale('log', basey=10)
plt.close(fig=2)
ax = plt.gca()
plt.ylabel('Nombre de cas confirmés', fontsize=15), plt.xlabel('')
plt.title('Evolution du COVID-19 en France, par région', fontsize=17,
          fontweight='bold')

# -----------------------------------------------------------------------------
# build vertical lines
date_range = np.arange(len(xlabs))
for d, t in vlines.items():
    idx = date_range[xlabs.index(d)]
    plt.axvline(idx, **vlines_kw)
    plt.text(idx, np.array(df_reg['vals']).max(), f"{t}", rotation=90,
             va='top', ha='right', fontsize=13, fontweight='bold')
# plt.show()
# exit()


###############################################################################
# Plotting France map
###############################################################################

# get latest covid and build the dict of data
covid_data = {}
for reg, dep in df_num_gp.items():
    for _dep in dep:
        covid_data[_dep] = late_covid.loc[reg]
# render the map
custom_style = Style(colors=('red', 'orange'), background='transparent',
                     plot_background='transparent')
fr_chart = pygal.maps.fr.Departments(style=custom_style)
fr_chart.add('COVID-19', covid_data)
fr_chart.render_to_png('img.png')
# plot the map
img = plt.imread('img.png')
ax = plt.subplot(gs[0, 1])
ax.imshow(img)
plt.tick_params(axis='both', which='both', bottom=False, left=False,
                top=False, labelbottom=False, labelleft=False)
ax.axis('off')
plt.title(f"COVID-19 {list(df['Date'])[-1]}")

plt.show()
