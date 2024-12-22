import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1: Import the data from medical_examination.csv and assign it to the df variable.
df = pd.read_csv('medical_examination.csv')

# 2: Add an overweight column to the data.
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3: Normalize cholesterol and glucose data.
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4: Draw the Categorical Plot.
def draw_cat_plot():
    # 5: Create a DataFrame for the categorical plot using pd.melt.
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6: Group and reformat the data to show the counts of each feature.
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat.rename(columns={'size': 'total'}, inplace=True)

    # 7: Draw the categorical plot using sns.catplot.
    catplot = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar',
        height=5,
        aspect=1
    )

    # 8: Get the figure for the output.
    fig = catplot.fig

    # 9: Save the plot.
    fig.savefig('catplot.png')
    return fig

# 10: Draw the Heat Map.
def draw_heat_map():
    # 11: Clean the data.
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) &
                 (df['height'] >= df['height'].quantile(0.025)) &
                 (df['height'] <= df['height'].quantile(0.975)) &
                 (df['weight'] >= df['weight'].quantile(0.025)) &
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12: Calculate the correlation matrix.
    corr = df_heat.corr()

    # 13: Generate a mask for the upper triangle.
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure.
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15: Draw the heatmap.
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        square=True,
        cmap='coolwarm',
        cbar_kws={'shrink': 0.5}
    )

    # 16: Save the heatmap.
    fig.savefig('heatmap.png')
    return fig
