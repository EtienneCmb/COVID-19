"""COVID-19 report for france."""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import pygal


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
# get latest covid and build the dict of data
late_covid = df.iloc[-1, :]
covid_data = {}
for reg, dep in df_num_gp.items():
    for _dep in dep:
        covid_data[_dep] = late_covid.loc[reg]


n_depart = len(depart)
# palette = sns.color_palette("hls", n_depart)
palette = sns.color_palette("tab20")




# ax.set_yscale('log', basey=10)

fr_chart = pygal.maps.fr.Departments()
fr_chart.title = f"COVID-19 {list(df['Date'])[-1]}"
fr_chart.add('COVID-19', covid_data)
fr_chart.render_to_png('img.png')


fig = plt.figure()
gs = fig.add_gridspec(2, 4)
ax1 = plt.subplot(gs[:, 0:2])
g = sns.factorplot(x="Date", y="vals", hue='Département', data=df_reg,
                   palette=palette, ax=ax1)
plt.close(fig=2)
img = plt.imread('img.png')
ax = plt.subplot(gs[0, 2])
ax.imshow(img)

# print(img)

plt.show()