import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

df = pd.read_csv('radar_points.csv')

plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)

scatter = ax.scatter(df['longitude'], df['latitude'],c=df['reflectivity'],cmap='viridis',transform=ccrs.PlateCarree(),s=50)

plt.colorbar(scatter, label='Reflectvity')

ax.set_extent([df['longitude'].min() - 0.5, 
               df['longitude'].max() + 0.5,
               df['latitude'].min() - 0.5, 
               df['latitude'].max() + 0.5])

# Add gridlines
ax.gridlines(draw_labels=True)

# Add title
plt.title('Radar Reflectivity Points')

# Save the plot
plt.savefig('radar_map.png', dpi=300, bbox_inches='tight')
plt.show()