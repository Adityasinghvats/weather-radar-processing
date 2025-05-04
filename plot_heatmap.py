import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

# Read the CSV file
df = pd.read_csv('radar_points.csv')

# Create a grid of points
grid_x, grid_y = np.mgrid[df['longitude'].min():df['longitude'].max():100j,
                         df['latitude'].min():df['latitude'].max():100j]

# Interpolate reflectivity values
grid_z = griddata((df['longitude'], df['latitude']), 
                 df['reflectivity'], 
                 (grid_x, grid_y), 
                 method='cubic')

# Create the plot
plt.figure(figsize=(12, 8))
plt.contourf(grid_x, grid_y, grid_z, cmap='viridis')
plt.colorbar(label='Reflectivity')

# Add points
plt.scatter(df['longitude'], df['latitude'], 
           c='red', s=1, alpha=0.5)

# Add labels and title
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Radar Reflectivity Heatmap')

# Save the plot
plt.savefig('radar_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()