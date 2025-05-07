import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

# Read the radar data
df = pd.read_csv('radar_points.csv')

# Create figure and axis with projection
fig = plt.figure(figsize=(15, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Add terrain background
ax.stock_img()  # This adds the Natural Earth background

# Add more detailed features
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax.add_feature(cfeature.BORDERS, linewidth=0.5)
ax.add_feature(cfeature.STATES, linewidth=0.3)

# Add terrain and land features
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS)

# Plot radar points
scatter = ax.scatter(df['longitude'], 
                    df['latitude'],
                    c=df['reflectivity'],
                    cmap='viridis',
                    transform=ccrs.PlateCarree(),
                    s=50,
                    alpha=0.6,  # Make points semi-transparent
                    edgecolor='black',
                    linewidth=0.5)

# Add colorbar
plt.colorbar(scatter, 
            label='Reflectivity (dBZ)',
            orientation='horizontal',
            pad=0.05)

# Set map extent with some padding
padding = 0.5
ax.set_extent([
    df['longitude'].min() - padding,
    df['longitude'].max() + padding,
    df['latitude'].min() - padding,
    df['latitude'].max() + padding
])

# Add gridlines
gl = ax.gridlines(draw_labels=True,
                 linewidth=0.5, 
                 color='gray', 
                 alpha=0.5,
                 linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Add title
plt.title('Radar Reflectivity Points Over Topography\n', pad=20)

# Add text box with radar site information
radar_text = f'Radar Site: ({df["longitude"].iloc[0]:.2f}°E, {df["latitude"].iloc[0]:.2f}°N)'
plt.text(0.02, 0.02, radar_text,
         transform=ax.transAxes,
         bbox=dict(facecolor='white', alpha=0.7),
         fontsize=8)

# Save high-resolution figure
plt.savefig('radar_topo_map.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.show()