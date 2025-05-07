import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
import geopandas as gpd
from shapely.geometry import Point

# Read the radar data
df = pd.read_csv('radar_points_2.csv')

# Convert DataFrame to GeoDataFrame
geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# Convert to Web Mercator projection (required for contextily)
gdf_web = gdf.to_crs(epsg=3857)

# Create the figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# First plot the points
scatter = ax.scatter(gdf_web.geometry.x, 
                    gdf_web.geometry.y,
                    c=gdf_web['reflectivity'],
                    cmap='viridis',
                    s=50,
                    alpha=0.6)

# Add colorbar
plt.colorbar(scatter, label='Reflectivity (dBZ)')

# Add the background map
ctx.add_basemap(ax, 
                source=ctx.providers.CartoDB.Positron,
                zoom='auto')  # Let contextily determine the zoom level

# Set extent to match the data
ax.set_xlim([gdf_web.geometry.x.min() - 1000, gdf_web.geometry.x.max() + 1000])
ax.set_ylim([gdf_web.geometry.y.min() - 1000, gdf_web.geometry.y.max() + 1000])

# Remove axes
ax.set_axis_off()

# Add title
plt.title('Radar Reflectivity Points', pad=20)

# Save the plot
plt.savefig('radar_street_map_2.png', 
            dpi=300, 
            bbox_inches='tight',
            pad_inches=0.1)

plt.show()