"""COVID-19 report for france."""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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
palette = sns.color_palette("Set3")



fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
g = sns.factorplot(x="Date", y="vals", hue='Département', data=df_reg,
                   palette=palette, ax=ax1)
ax = plt.gca()
# ax.set_yscale('log', basey=10)

plt.sca(ax2)
fr_chart = pygal.maps.fr.Departments()
fr_chart.title = f"COVID-19 {list(df['Date'])[-1]}"
fr_chart.add('COVID-19', covid_data)
fr_chart.render_to_png('img.png')
img = plt.imread('img.png')
plt.imshow(img)

# print(img)

plt.show()