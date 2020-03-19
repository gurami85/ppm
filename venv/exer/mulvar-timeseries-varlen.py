import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("../data/sales1.csv")
df2 = pd.read_csv("../data/sales2.csv")
print('---- df1 ----')
print(df1)
print('---- df2 ----')
print(df2)

# merge all items of time from two data frames (can cause duplicates)
time_list_row = []
for idx, row in df1.iterrows():
    time_list_row.append(row['Time'])

for idx, row in df2.iterrows():
    time_list_row.append(row['Time'])

# make a new list without duplicates
time_list = []
for i in time_list_row:
    if i not in time_list:
        time_list.append(i)

# sort the list
time_list.sort()
print(*time_list, sep='\n')

# make a result data frame with default values (0)
df_total = pd.DataFrame(time_list, columns=['Time'])
df_total['Sales1'] = 0
df_total['Sales2'] = 0

# fill the data frame with the values from each data frame
for idx, row in df1.iterrows():
    df_total.loc[df_total.Time == row['Time'], 'Sales1'] = row['Sales']

for idx, row in df2.iterrows():
    df_total.loc[df_total.Time == row['Time'], 'Sales2'] = row['Sales']
print(df_total)

# extend the values not 0
base = 0
for idx, row in df_total.iterrows():
    if row['Sales1'] == 0:
        df_total.loc[df_total.Time == row['Time'], 'Sales1'] = base
    else:
        base = row['Sales1']

base = 0
for idx, row in df_total.iterrows():
    if row['Sales2'] == 0:
        df_total.loc[df_total.Time == row['Time'], 'Sales2'] = base
    else:
        base = row['Sales2']
print(df_total)

# plot style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')

# multiple line plot
num = 0
for column in df_total.drop('Time', axis=1):
    plt.plot(df_total['Time'], df_total[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
    num += 1

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("Line plot for sales", loc='left', fontsize=12, fontweight=0, color='black')
plt.xlabel("Time")
plt.ylabel("Sales")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

plt.show()