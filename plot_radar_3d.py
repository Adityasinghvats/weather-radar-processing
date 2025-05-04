import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('radar_points.csv')

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

scatter = ax.scatter(df['longitude'], 
                    df['latitude'], 
                    df['altitude'],
                    c=df['reflectivity'],
                    cmap='viridis')

# Add labels
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Altitude (m)')

# Add colorbar
plt.colorbar(scatter, label='Reflectivity')

# Add title
plt.title('3D Radar Reflectivity Points')

# Save the plot
plt.savefig('radar_3d.png', dpi=300, bbox_inches='tight')
plt.show()