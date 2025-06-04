import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

class RadarRainfallVisualizer:
    def __init__(self, csv_file):
        """
        Initialize the visualizer with radar rainfall CSV data
        """
        self.csv_file = csv_file
        self.load_data()
        
    def load_data(self):
        """Load and prepare the data"""
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"CSV file '{self.csv_file}' not found")
        
        print(f"Loading data from {self.csv_file}...")
        self.df = pd.read_csv(self.csv_file)
        
        print(f"Loaded {len(self.df):,} data points")
        print(f"Columns: {list(self.df.columns)}")
        print(f"Data range:")
        print(f"  Latitude: {self.df['latitude'].min():.4f} to {self.df['latitude'].max():.4f}")
        print(f"  Longitude: {self.df['longitude'].min():.4f} to {self.df['longitude'].max():.4f}")
        print(f"  Reflectivity: {self.df['reflectivity_dbz'].min():.1f} to {self.df['reflectivity_dbz'].max():.1f} dBZ")
        print(f"  Rainfall: {self.df['rainfall_rate_mm_hr'].min():.3f} to {self.df['rainfall_rate_mm_hr'].max():.3f} mm/hr")
    
    def create_cartopy_maps(self):
        """Create publication-quality maps using Cartopy"""
        print("Creating Cartopy geographic maps...")
        
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
        except ImportError:
            print("Cartopy not available. Skipping geographic maps.")
            return
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(20, 12))
        
        # Define map extent
        lat_min, lat_max = self.df['latitude'].min() - 0.1, self.df['latitude'].max() + 0.1
        lon_min, lon_max = self.df['longitude'].min() - 0.1, self.df['longitude'].max() + 0.1
        
        # 1. Reflectivity map
        ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax1.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax1.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)
        ax1.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax1.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot reflectivity data
        scatter1 = ax1.scatter(self.df['longitude'], self.df['latitude'], 
                              c=self.df['reflectivity_dbz'], 
                              cmap='plasma', s=2, alpha=0.7,
                              transform=ccrs.PlateCarree())
        
        # Add radar location
        ax1.plot(88.3697, 22.5697, 'r*', markersize=20, 
                transform=ccrs.PlateCarree(), label='Kolkata Radar')
        
        # Add Alipore Observatory
        ax1.plot(88.3333, 22.5333, 'ko', markersize=12, 
                transform=ccrs.PlateCarree(), label='Alipore Observatory')
        
        # Add gridlines
        ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax1.set_title('Radar Reflectivity (dBZ)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, orientation='horizontal', 
                            pad=0.1, shrink=0.8)
        cbar1.set_label('Reflectivity (dBZ)', fontsize=12)
        
        # 2. Rainfall map
        ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
        
        # Add map features
        ax2.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax2.add_feature(cfeature.RIVERS, linewidth=0.3, alpha=0.5)
        ax2.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax2.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot rainfall data
        scatter2 = ax2.scatter(self.df['longitude'], self.df['latitude'], 
                              c=self.df['rainfall_rate_mm_hr'], 
                              cmap='Blues', s=2, alpha=0.7,
                              transform=ccrs.PlateCarree())
        
        # Add radar location
        ax2.plot(88.3697, 22.5697, 'r*', markersize=20, 
                transform=ccrs.PlateCarree(), label='Kolkata Radar')
        
        # Add Alipore Observatory
        ax2.plot(88.3333, 22.5333, 'ko', markersize=12, 
                transform=ccrs.PlateCarree(), label='Alipore Observatory')
        
        # Add gridlines
        ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        ax2.set_title('Rainfall Rate (mm/hr)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2, orientation='horizontal', 
                            pad=0.1, shrink=0.8)
        cbar2.set_label('Rainfall Rate (mm/hr)', fontsize=12)
        
        plt.tight_layout()
        
        # Save the maps
        output_file = self.csv_file.replace('.csv', '_cartopy_maps.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Cartopy maps saved as: {output_file}")
        plt.show()
    
    def create_interpolated_maps(self):
        """Create interpolated contour maps"""
        print("Creating interpolated contour maps...")
        
        # Create grid for interpolation
        lat_min, lat_max = self.df['latitude'].min(), self.df['latitude'].max()
        lon_min, lon_max = self.df['longitude'].min(), self.df['longitude'].max()
        
        # Create regular grid
        grid_resolution = 0.01  # degrees
        lons = np.arange(lon_min, lon_max, grid_resolution)
        lats = np.arange(lat_min, lat_max, grid_resolution)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        
        # Sample data for interpolation (to avoid memory issues)
        sample_size = min(5000, len(self.df))
        sample_df = self.df.sample(n=sample_size)
        
        # Interpolate reflectivity
        points = np.column_stack((sample_df['longitude'], sample_df['latitude']))
        refl_grid = griddata(points, sample_df['reflectivity_dbz'], 
                            (lon_grid, lat_grid), method='linear')
        
        # Interpolate rainfall
        rain_grid = griddata(points, sample_df['rainfall_rate_mm_hr'], 
                            (lon_grid, lat_grid), method='linear')
        
        # Create figure with interpolated maps
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Reflectivity contour map
        contour1 = ax1.contourf(lon_grid, lat_grid, refl_grid, levels=20, cmap='plasma')
        ax1.contour(lon_grid, lat_grid, refl_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax1.scatter(sample_df['longitude'], sample_df['latitude'], 
                   c='white', s=0.5, alpha=0.7)
        ax1.plot(88.3697, 22.5697, 'r*', markersize=15, label='Kolkata Radar')
        ax1.plot(88.3333, 22.5333, 'ko', markersize=10, label='Alipore Observatory')
        ax1.set_xlabel('Longitude (°E)')
        ax1.set_ylabel('Latitude (°N)')
        ax1.set_title('Interpolated Reflectivity (dBZ)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add colorbar
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label('Reflectivity (dBZ)')
        
        # Rainfall contour map
        contour2 = ax2.contourf(lon_grid, lat_grid, rain_grid, levels=20, cmap='Blues')
        ax2.contour(lon_grid, lat_grid, rain_grid, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        ax2.scatter(sample_df['longitude'], sample_df['latitude'], 
                   c='white', s=0.5, alpha=0.7)
        ax2.plot(88.3697, 22.5697, 'r*', markersize=15, label='Kolkata Radar')
        ax2.plot(88.3333, 22.5333, 'ko', markersize=10, label='Alipore Observatory')
        ax2.set_xlabel('Longitude (°E)')
        ax2.set_ylabel('Latitude (°N)')
        ax2.set_title('Interpolated Rainfall Rate (mm/hr)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add colorbar
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Rainfall Rate (mm/hr)')
        
        plt.tight_layout()
        
        # Save interpolated maps
        output_file = self.csv_file.replace('.csv', '_interpolated_maps.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Interpolated maps saved as: {output_file}")
        plt.show()
    
    def create_static_plots(self):
        """Create static matplotlib/seaborn plots"""
        print("Creating static plots...")
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("viridis")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Reflectivity map
        ax1 = plt.subplot(2, 3, 1)
        scatter1 = ax1.scatter(self.df['longitude'], self.df['latitude'], 
                              c=self.df['reflectivity_dbz'], 
                              cmap='plasma', s=1, alpha=0.6)
        ax1.set_xlabel('Longitude (°E)')
        ax1.set_ylabel('Latitude (°N)')
        ax1.set_title('Radar Reflectivity (dBZ)')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=ax1, label='Reflectivity (dBZ)')
        
        # Add radar location
        radar_lat, radar_lon = 22.5697, 88.3697
        ax1.plot(radar_lon, radar_lat, 'r*', markersize=15, label='Kolkata Radar')
        
        # Add Alipore Observatory
        alipore_lat, alipore_lon = 22.5333, 88.3333
        ax1.plot(alipore_lon, alipore_lat, 'ko', markersize=8, label='Alipore Observatory')
        ax1.legend()
        
        # 2. Rainfall rate map
        ax2 = plt.subplot(2, 3, 2)
        scatter2 = ax2.scatter(self.df['longitude'], self.df['latitude'], 
                              c=self.df['rainfall_rate_mm_hr'], 
                              cmap='Blues', s=1, alpha=0.6)
        ax2.set_xlabel('Longitude (°E)')
        ax2.set_ylabel('Latitude (°N)')
        ax2.set_title('Rainfall Rate (mm/hr)')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Rainfall Rate (mm/hr)')
        
        # Add radar location
        ax2.plot(radar_lon, radar_lat, 'r*', markersize=15, label='Kolkata Radar')
        ax2.plot(alipore_lon, alipore_lat, 'ko', markersize=8, label='Alipore Observatory')
        ax2.legend()
        
        # 3. Reflectivity histogram
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(self.df['reflectivity_dbz'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.set_xlabel('Reflectivity (dBZ)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Reflectivity Distribution')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(self.df['reflectivity_dbz'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {self.df["reflectivity_dbz"].mean():.1f} dBZ')
        ax3.legend()
        
        # 4. Rainfall rate histogram
        ax4 = plt.subplot(2, 3, 4)
        ax4.hist(self.df['rainfall_rate_mm_hr'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.set_xlabel('Rainfall Rate (mm/hr)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Rainfall Rate Distribution')
        ax4.grid(True, alpha=0.3)
        ax4.axvline(self.df['rainfall_rate_mm_hr'].mean(), color='red', linestyle='--',
                   label=f'Mean: {self.df["rainfall_rate_mm_hr"].mean():.2f} mm/hr')
        ax4.legend()
        
        # 5. Reflectivity vs Rainfall scatter plot
        ax5 = plt.subplot(2, 3, 5)
        ax5.scatter(self.df['reflectivity_dbz'], self.df['rainfall_rate_mm_hr'], 
                   alpha=0.5, s=1, c='purple')
        ax5.set_xlabel('Reflectivity (dBZ)')
        ax5.set_ylabel('Rainfall Rate (mm/hr)')
        ax5.set_title('Z-R Relationship')
        ax5.grid(True, alpha=0.3)
        
        # Add theoretical Z-R curve (Marshall-Palmer)
        z_range = np.linspace(0, 60, 100)
        z_linear = 10**(z_range/10)
        r_theoretical = (z_linear/200)**(1/1.6)
        ax5.plot(z_range, r_theoretical, 'r-', linewidth=2, label='Marshall-Palmer')
        ax5.legend()
        
        # 6. Rainfall intensity categories
        ax6 = plt.subplot(2, 3, 6)
        if 'rainfall_intensity_category' in self.df.columns:
            intensity_counts = self.df['rainfall_intensity_category'].value_counts()
        else:
            categories = self.categorize_rainfall(self.df['rainfall_rate_mm_hr'])
            intensity_counts = pd.Series(categories).value_counts()
        
        colors = ['lightblue', 'yellow', 'orange', 'red', 'darkred']
        ax6.pie(intensity_counts.values, labels=intensity_counts.index, 
               autopct='%1.1f%%', colors=colors[:len(intensity_counts)])
        ax6.set_title('Rainfall Intensity Distribution')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.csv_file.replace('.csv', '_static_plots.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Static plots saved as: {output_file}")
        plt.show()
    
    def create_interactive_plots(self):
        """Create interactive Plotly visualizations"""
        print("Creating interactive plots...")
        
        # 1. Interactive reflectivity map
        fig_refl = px.scatter_mapbox(
            self.df.sample(n=min(10000, len(self.df))),  # Sample for performance
            lat='latitude',
            lon='longitude',
            color='reflectivity_dbz',
            size='rainfall_rate_mm_hr',
            hover_data=['reflectivity_dbz', 'rainfall_rate_mm_hr'],
            color_continuous_scale='Plasma',
            title='Interactive Radar Reflectivity Map',
            mapbox_style='open-street-map',
            zoom=9,
            height=600
        )
        
        # Add radar location
        fig_refl.add_trace(go.Scattermapbox(
            lat=[22.5697],
            lon=[88.3697],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Kolkata Radar',
            hovertext='Kolkata Weather Radar'
        ))
        
        # Add Alipore Observatory
        fig_refl.add_trace(go.Scattermapbox(
            lat=[22.5333],
            lon=[88.3333],
            mode='markers',
            marker=dict(size=15, color='black', symbol='circle'),
            name='Alipore Observatory',
            hovertext='Alipore Observatory'
        ))
        
        fig_refl.update_layout(mapbox_center={"lat": 22.55, "lon": 88.35})
        
        # Save interactive reflectivity map
        refl_html = self.csv_file.replace('.csv', '_reflectivity_map.html')
        fig_refl.write_html(refl_html)
        print(f"Interactive reflectivity map saved as: {refl_html}")
        fig_refl.show()
        
        # 2. Interactive rainfall map
        fig_rain = px.scatter_mapbox(
            self.df.sample(n=min(10000, len(self.df))),  # Sample for performance
            lat='latitude',
            lon='longitude',
            color='rainfall_rate_mm_hr',
            size='reflectivity_dbz',
            hover_data=['reflectivity_dbz', 'rainfall_rate_mm_hr'],
            color_continuous_scale='Blues',
            title='Interactive Rainfall Rate Map',
            mapbox_style='open-street-map',
            zoom=9,
            height=600
        )
        
        # Add radar location
        fig_rain.add_trace(go.Scattermapbox(
            lat=[22.5697],
            lon=[88.3697],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star'),
            name='Kolkata Radar',
            hovertext='Kolkata Weather Radar'
        ))
        
        # Add Alipore Observatory
        fig_rain.add_trace(go.Scattermapbox(
            lat=[22.5333],
            lon=[88.3333],
            mode='markers',
            marker=dict(size=15, color='black', symbol='circle'),
            name='Alipore Observatory',
            hovertext='Alipore Observatory'
        ))
        
        fig_rain.update_layout(mapbox_center={"lat": 22.55, "lon": 88.35})
        
        # Save interactive rainfall map
        rain_html = self.csv_file.replace('.csv', '_rainfall_map.html')
        fig_rain.write_html(rain_html)
        print(f"Interactive rainfall map saved as: {rain_html}")
        fig_rain.show()
        
        # 3. Combined dashboard
        fig_dashboard = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Reflectivity vs Rainfall', 'Reflectivity Distribution', 
                           'Rainfall Distribution', 'Distance from Radar'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Z-R relationship
        fig_dashboard.add_trace(
            go.Scatter(x=self.df['reflectivity_dbz'], 
                      y=self.df['rainfall_rate_mm_hr'],
                      mode='markers',
                      marker=dict(size=3, opacity=0.6),
                      name='Data Points'),
            row=1, col=1
        )
        
        # Theoretical Z-R curve
        z_range = np.linspace(0, 60, 100)
        z_linear = 10**(z_range/10)
        r_theoretical = (z_linear/200)**(1/1.6)
        fig_dashboard.add_trace(
            go.Scatter(x=z_range, y=r_theoretical,
                      mode='lines',
                      line=dict(color='red', width=3),
                      name='Marshall-Palmer'),
            row=1, col=1
        )
        
        # Reflectivity histogram
        fig_dashboard.add_trace(
            go.Histogram(x=self.df['reflectivity_dbz'], 
                        name='Reflectivity', 
                        nbinsx=50),
            row=1, col=2
        )
        
        # Rainfall histogram
        fig_dashboard.add_trace(
            go.Histogram(x=self.df['rainfall_rate_mm_hr'], 
                        name='Rainfall Rate',
                        nbinsx=50),
            row=2, col=1
        )
        
        # Distance from radar
        radar_lat, radar_lon = 22.5697, 88.3697
        distances = np.sqrt((self.df['latitude'] - radar_lat)**2 + 
                           (self.df['longitude'] - radar_lon)**2) * 111  # Convert to km
        
        fig_dashboard.add_trace(
            go.Scatter(x=distances,
                      y=self.df['rainfall_rate_mm_hr'],
                      mode='markers',
                      marker=dict(size=3, opacity=0.6),
                      name='Distance vs Rainfall'),
            row=2, col=2
        )
        
        fig_dashboard.update_layout(height=800, showlegend=False,
                                   title_text="Radar Data Analysis Dashboard")
        
        # Save dashboard
        dashboard_html = self.csv_file.replace('.csv', '_dashboard.html')
        fig_dashboard.write_html(dashboard_html)
        print(f"Interactive dashboard saved as: {dashboard_html}")
        fig_dashboard.show()
    
    def create_folium_map(self):
        """Create advanced Folium maps"""
        print("Creating Folium maps...")
        
        # Center the map on Kolkata
        center_lat = self.df['latitude'].mean()
        center_lon = self.df['longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add different tile layers
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('Stamen Toner').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Add radar location
        folium.Marker(
            [22.5697, 88.3697],
            popup=folium.Popup('Kolkata Weather Radar<br>Location: 22.5697°N, 88.3697°E', max_width=300),
            icon=folium.Icon(color='red', icon='star'),
            tooltip='Kolkata Radar'
        ).add_to(m)
        
        # Add Alipore Observatory
        folium.Marker(
            [22.5333, 88.3333],
            popup=folium.Popup('Alipore Observatory<br>Location: 22.5333°N, 88.3333°E', max_width=300),
            icon=folium.Icon(color='black', icon='info-sign'),
            tooltip='Alipore Observatory'
        ).add_to(m)
        
        # Sample data for different visualizations
        sample_size = min(5000, len(self.df))
        sample_df = self.df.sample(n=sample_size)
        
        # Create rainfall heat map data
        heat_data = []
        for idx, row in sample_df.iterrows():
            if row['rainfall_rate_mm_hr'] > 0:  # Only show areas with rainfall
                heat_data.append([row['latitude'], row['longitude'], row['rainfall_rate_mm_hr']])
        
        # Add rainfall heat map
        if heat_data:
            HeatMap(
                heat_data, 
                radius=15, 
                blur=10, 
                gradient={
                    0.0: 'blue',
                    0.3: 'cyan',
                    0.5: 'lime', 
                    0.7: 'yellow',
                    0.9: 'orange',
                    1.0: 'red'
                },
                name='Rainfall Heat Map'
            ).add_to(m)
        
        # Add marker cluster for high rainfall areas
        high_rainfall = sample_df[sample_df['rainfall_rate_mm_hr'] > 5]
        if len(high_rainfall) > 0:
            marker_cluster = MarkerCluster(name='High Rainfall Areas (>5 mm/hr)')
            
            for idx, row in high_rainfall.iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=5,
                    popup=f"Rainfall: {row['rainfall_rate_mm_hr']:.2f} mm/hr<br>Reflectivity: {row['reflectivity_dbz']:.1f} dBZ",
                    color='red',
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(marker_cluster)
            
            marker_cluster.add_to(m)
        
        # Add range rings around radar
        for radius in [50, 100, 150, 200]:  # km
            folium.Circle(
                location=[22.5697, 88.3697],
                radius=radius * 1000,  # Convert to meters
                color='gray',
                weight=1,
                fill=False,
                opacity=0.5,
                popup=f'{radius} km range'
            ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Save map
        map_html = self.csv_file.replace('.csv', '_folium_map.html')
        m.save(map_html)
        print(f"Folium map saved as: {map_html}")
        
        return m
    
    def create_radar_ppi_plot(self):
        """Create Plan Position Indicator (PPI) plot like traditional radar displays"""
        print("Creating radar PPI plot...")
        
        # Calculate polar coordinates relative to radar
        radar_lat, radar_lon = 22.5697, 88.3697
        
        # Convert to relative coordinates (approximately)
        lat_diff = (self.df['latitude'] - radar_lat) * 111  # km
        lon_diff = (self.df['longitude'] - radar_lon) * 111 * np.cos(np.radians(radar_lat))  # km
        
        # Calculate range and azimuth
        ranges = np.sqrt(lat_diff**2 + lon_diff**2)
        azimuths = np.degrees(np.arctan2(lon_diff, lat_diff))
        azimuths[azimuths < 0] += 360  # Convert to 0-360 degrees
        
        # Create PPI plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9), subplot_kw=dict(projection='polar'))
        
        # Convert azimuth to radians for polar plot
        theta = np.radians(azimuths)
        
        # 1. Reflectivity PPI
        scatter1 = ax1.scatter(theta, ranges, c=self.df['reflectivity_dbz'], 
                              cmap='plasma', s=2, alpha=0.7)
        ax1.set_title('Reflectivity PPI (dBZ)', pad=20, fontsize=14, fontweight='bold')
        ax1.set_theta_zero_location('N')  # North at top
        ax1.set_theta_direction(-1)  # Clockwise
        ax1.set_ylim(0, 250)  # 250 km range
        ax1.set_ylabel('Range (km)', labelpad=30)
        
        # Add range rings
        for r in [50, 100, 150, 200]:
            circle = plt.Circle((0, 0), r, fill=False, color='gray', alpha=0.5, linewidth=0.5)
            ax1.add_patch(circle)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, pad=0.1, shrink=0.8)
        cbar1.set_label('Reflectivity (dBZ)', fontsize=12)
        
        # 2. Rainfall PPI
        scatter2 = ax2.scatter(theta, ranges, c=self.df['rainfall_rate_mm_hr'], 
                              cmap='Blues', s=2, alpha=0.7)
        ax2.set_title('Rainfall Rate PPI (mm/hr)', pad=20, fontsize=14, fontweight='bold')
        ax2.set_theta_zero_location('N')  # North at top
        ax2.set_theta_direction(-1)  # Clockwise
        ax2.set_ylim(0, 250)  # 250 km range
        ax2.set_ylabel('Range (km)', labelpad=30)
        
        # Add range rings
        for r in [50, 100, 150, 200]:
            circle = plt.Circle((0, 0), r, fill=False, color='gray', alpha=0.5, linewidth=0.5)
            ax2.add_patch(circle)
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2, pad=0.1, shrink=0.8)
        cbar2.set_label('Rainfall Rate (mm/hr)', fontsize=12)
        
        plt.tight_layout()
        
        # Save PPI plot
        output_file = self.csv_file.replace('.csv', '_ppi_plot.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"PPI plot saved as: {output_file}")
        plt.show()
    
    def categorize_rainfall(self, rainfall_rates):
        """Categorize rainfall intensity"""
        categories = []
        for rate in rainfall_rates:
            if rate <= 0:
                categories.append('No Rain')
            elif rate <= 2.5:
                categories.append('Light')
            elif rate <= 10.0:
                categories.append('Moderate')
            elif rate <= 50.0:
                categories.append('Heavy')
            else:
                categories.append('Very Heavy')
        return categories
    
    def print_statistics(self):
        """Print detailed statistics"""
        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)
        
        print(f"\nDataset Overview:")
        print(f"  Total data points: {len(self.df):,}")
        print(f"  Geographic coverage:")
        print(f"    Latitude: {self.df['latitude'].min():.4f}° to {self.df['latitude'].max():.4f}°N")
        print(f"    Longitude: {self.df['longitude'].min():.4f}° to {self.df['longitude'].max():.4f}°E")
        
        print(f"\nReflectivity Statistics:")
        print(f"  Min: {self.df['reflectivity_dbz'].min():.1f} dBZ")
        print(f"  Max: {self.df['reflectivity_dbz'].max():.1f} dBZ")
        print(f"  Mean: {self.df['reflectivity_dbz'].mean():.1f} dBZ")
        print(f"  Median: {self.df['reflectivity_dbz'].median():.1f} dBZ")
        print(f"  Std Dev: {self.df['reflectivity_dbz'].std():.1f} dBZ")
        
        print(f"\nRainfall Rate Statistics:")
        print(f"  Min: {self.df['rainfall_rate_mm_hr'].min():.3f} mm/hr")
        print(f"  Max: {self.df['rainfall_rate_mm_hr'].max():.3f} mm/hr")
        print(f"  Mean: {self.df['rainfall_rate_mm_hr'].mean():.3f} mm/hr")
        print(f"  Median: {self.df['rainfall_rate_mm_hr'].median():.3f} mm/hr")
        print(f"  Std Dev: {self.df['rainfall_rate_mm_hr'].std():.3f} mm/hr")
        
        # Rainfall intensity distribution
        if 'rainfall_intensity_category' in self.df.columns:
            intensity_counts = self.df['rainfall_intensity_category'].value_counts()
        else:
            categories = self.categorize_rainfall(self.df['rainfall_rate_mm_hr'])
            intensity_counts = pd.Series(categories).value_counts()
        
        print(f"\nRainfall Intensity Distribution:")
        for category, count in intensity_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {category}: {count:,} points ({percentage:.1f}%)")

def main():
    """Main function to create visualizations"""
    
    # List of possible CSV files
    csv_files = [
        'kolkata_radar_rainfall_tropical_lowest_elevation.csv',
        'kolkata_radar_rainfall_marshall_palmer_lowest_elevation.csv',
        'kolkata_radar_rainfall_tropical_filtered.csv',
        'kolkata_radar_rainfall_tropical.csv'
    ]
    
    # Find the first available CSV file
    csv_file = None
    for file in csv_files:
        if os.path.exists(file):
            csv_file = file
            break
    
    if csv_file is None:
        print("No CSV files found. Please run the radar processing script first.")
        print("Looking for files:")
        for file in csv_files:
            print(f"  - {file}")
        return
    
    print(f"Using CSV file: {csv_file}")
    
    try:
        # Create visualizer
        visualizer = RadarRainfallVisualizer(csv_file)
        
        # Print statistics
        visualizer.print_statistics()
        
        # Create all visualizations
        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        
        # Static plots
        visualizer.create_static_plots()
        
        # Geographic maps with Cartopy (if available)
        try:
            visualizer.create_cartopy_maps()
        except ImportError:
            print("Cartopy not installed. Skipping geographic maps.")
            print("Install with: pip install cartopy")
        
        # Interpolated contour maps
        visualizer.create_interpolated_maps()
        
        # Traditional radar PPI plot
        visualizer.create_radar_ppi_plot()
        
        # Interactive plots
        visualizer.create_interactive_plots()
        
        # Folium maps
        visualizer.create_folium_map()
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE")
        print("="*60)
        print("Generated files:")
        base_name = csv_file.replace('.csv', '')
        print(f"  • {base_name}_static_plots.png")
        print(f"  • {base_name}_cartopy_maps.png")
        print(f"  • {base_name}_interpolated_maps.png")
        print(f"  • {base_name}_ppi_plot.png")
        print(f"  • {base_name}_reflectivity_map.html")
        print(f"  • {base_name}_rainfall_map.html")
        print(f"  • {base_name}_dashboard.html")
        print(f"  • {base_name}_folium_map.html")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()