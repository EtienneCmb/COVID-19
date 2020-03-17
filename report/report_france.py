"""COVID-19 report for france.

COVID-19 data for France :
https://www.data.gouv.fr/en/datasets/cas-confirmes-dinfection-au-covid-19-par-region/
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import pygal
from pygal.style import BlueStyle, Style

sns.set_style("whitegrid")
plt.rc('font', family="serif")

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
# date conversion and formating
df['Date'] = pd.to_datetime(list(df['Date'])).strftime('%d/%m')
df_reg = df.melt('Date', var_name='Département', value_name='vals')
depart = np.unique(df_reg['Département'])
n_depart = len(depart)
palette = sns.color_palette("tab20")


###############################################################################
# Plotting time-series
###############################################################################

fig = plt.figure()
gs = fig.add_gridspec(1, 2)
ax1 = plt.subplot(gs[0, 0])
g = sns.factorplot(x="Date", y="vals", hue='Département', data=df_reg,
                   palette=palette, ax=ax1)
# ax1.set_yscale('log', basey=10)
plt.close(fig=2)


###############################################################################
# Plotting France map
###############################################################################

# get latest covid and build the dict of data
late_covid = df.iloc[-1, :]
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
