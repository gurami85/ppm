import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/monthly-sunspots.csv")

# plot style
plt.style.use('seaborn-darkgrid')

# create a color palette
palette = plt.get_cmap('Set1')
plt.plot(df['Month'], df['Sunspots'], marker='', color=palette(0), linewidth=1, alpha=0.9, label='Sunspots')

# set the number of ticks
ax1 = plt.gca()
ax1.xaxis.set_major_locator(plt.LinearLocator(numticks=10))

# Add legend
plt.legend(loc=2, ncol=2)

# Add titles
plt.title("Monthly Sunspots", loc='left', fontsize=12, fontweight=0, color='black')
plt.xlabel("Month")
plt.ylabel("Num. of sunspots")

plt.show()