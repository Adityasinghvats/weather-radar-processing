# Weather Radar Data Processing System

A comprehensive Python toolkit for processing weather radar data from VOL files to NetCDF format and generating rainfall estimates for the Kolkata region.

## Overview

This system processes raw radar volume files (.vol) from the India Meteorological Department's Doppler Weather Radar and converts them into standardized NetCDF format for analysis and visualization. It includes tools for:

- Converting VOL files to NetCDF format
- Processing radar data to extract rainfall estimates
- Creating visualizations and maps
- Generating CSV and JSON outputs for further analysis

## System Requirements

- Python 3.8 or higher
- Git
- Windows/Linux/macOS
- At least 4GB RAM (8GB recommended for large datasets)
- 1GB free disk space

## Installation Guide

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/yourusername/weather-radar-processing.git
cd weather-radar-processing

# Alternative: If you have the code as a ZIP file
# Extract the ZIP file and navigate to the directory
```

### 2. Set Up Python Virtual Environment

#### On Windows:
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Verify activation (you should see (.venv) in your prompt)
```

#### On Linux/macOS:
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (you should see (.venv) in your prompt)
```

### 3. Install Dependencies

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt

# If requirements.txt doesn't exist, install manually:
pip install numpy pandas xarray netcdf4 matplotlib seaborn plotly folium cartopy scipy wradlib
```

### 4. Create requirements.txt (if not provided)

```bash
# Generate requirements.txt for future use
pip freeze > requirements.txt
```

## Required Dependencies

Create a `requirements.txt` file with the following content:

```txt
numpy>=1.21.0
pandas>=1.3.0
xarray>=0.19.0
netcdf4>=1.5.7
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
folium>=0.12.0
cartopy>=0.20.0
scipy>=1.7.0
wradlib>=1.19.0
```

## Project Structure

```
weather-radar-processing/
├── README.md
├── requirements.txt
├── voltonc.py          # VOL to NetCDF converter
├── processnc.py        # NetCDF processor for rainfall estimation
├── findcord.py         # Coordinate extraction utility
├── visualization.py    # Visualization tools
├── data/              # Raw radar data directory
│   ├── *.vol          # Raw VOL files
│   └── *.nc           # Processed NetCDF files
├── output/            # Generated outputs
│   ├── *.csv          # CSV data files
│   ├── *.json         # JSON data files
│   └── *.png          # Visualization images
└── .venv/             # Virtual environment
```

## Usage Guide

### Step 1: Convert VOL Files to NetCDF

Place your radar VOL files in the project directory and run:

```bash
# Ensure virtual environment is activated
# (.venv) should be visible in your command prompt

# Convert VOL files to NetCDF
python voltonc.py
```

**Expected VOL file naming convention:**
- `2025042623000000V.vol` - Velocity data
- `2025042623000000W.vol` - Spectrum width data  
- `2025042623000000dBZ.vol` - Reflectivity data

**What this does:**
- Parses binary VOL files
- Extracts radar reflectivity data
- Converts to geographic coordinates
- Saves as NetCDF file (e.g., `2025042623000000.nc`)

### Step 2: Process NetCDF Data for Rainfall Estimation

```bash
# Process NetCDF to generate rainfall estimates
python processnc.py
```

**What this does:**
- **You will need NetCDF files to upload to our backend endpoint for getting the data for plotting**.
- Reads NetCDF radar data
- Selects lowest elevation sweep (closest to ground)
- Applies Marshall-Palmer Z-R relationship
- Calculates rainfall rates
- Generates CSV output with coordinates and rainfall data

## Configuration

### Radar Location Settings

Edit the radar coordinates in `voltonc.py`:

```python
# Kolkata radar coordinates (default)
processor = RadarVolumeProcessor(
    radar_lat=22.5697,    # Latitude in degrees
    radar_lon=88.3697,    # Longitude in degrees  
    max_range_km=250      # Maximum radar range in km
)
```

### Z-R Relationship Selection

Modify the Z-R relationship in `processnc.py`:

```python
# Available options:
# 'marshall_palmer' - Standard relationship
# 'tropical' - For tropical/monsoon conditions (recommended for India)
# 'convective' - For convective storms
# 'stratiform' - For stratiform rain

zr_relation = 'tropical'  # Change this as needed
```

### Reflectivity Thresholds

Adjust filtering thresholds in `processnc.py`:

```python
df = processor.process_to_dataframe(
    min_reflectivity=0,      # Minimum reflectivity (dBZ)
    max_reflectivity=60,     # Maximum reflectivity (dBZ)
    zr_relation='tropical'
)
```

## Output Files

### CSV Files
- `kolkata2_radar_rainfall_tropical_lowest_elevation.csv` - Main data output
- `*_summary.csv` - Statistical summary by rainfall intensity

### JSON Files
- `*.json` - Structured data for web applications
- `*.geojson` - Geographic data for mapping

### Visualization Files
- `*_static_plots.png` - Static charts and maps
- `*_interactive_map.html` - Interactive web maps
- `*_dashboard.html` - Analysis dashboard

## Data Format

### CSV Output Structure
```csv
latitude,longitude,altitude,reflectivity_dbz,rainfall_rate_mm_hr,sweep_number,elevation_angle,rainfall_intensity_category,timestamp
22.5697,88.3697,45.2,25.3,2.15,0,0.5,Light,2025-04-26T23:00:00
```

### JSON Output Structure
```json
[
  {
    "latitude": 22.5697,
    "longitude": 88.3697,
    "reflectivity": 25.3,
    "rainfall_rate": 2.15
  }
]
```

## Troubleshooting

### Common Issues

1. **"No module named 'wradlib'"**
   ```bash
   pip install wradlib
   ```

2. **"NetCDF file not found"**
   - Ensure VOL files are in the project directory
   - Run `voltonc.py` first to create NetCDF files

3. **"Memory Error" with large files**
   - Reduce `max_range_km` in configuration
   - Process files individually rather than in batches

4. **Virtual environment not activated**
   ```bash
   # Windows
   .venv\Scripts\activate
   
   # Linux/macOS
   source .venv/bin/activate
   ```

5. **Cartopy installation issues**
   ```bash
   # If cartopy fails to install
   pip install cartopy --no-binary cartopy
   ```

### Performance Tips

- Process one timestamp at a time for large datasets
- Reduce spatial resolution if memory is limited
- Use `min_reflectivity` filtering to reduce data size

## Data Sources

- **Radar Data**: India Meteorological Department (IMD)
- **Location**: Kolkata Doppler Weather Radar
- **Coordinates**: 22.5697°N, 88.3697°E
- **Coverage**: 250 km radius around Kolkata

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For issues and questions:
- Check the troubleshooting section above
- Create an issue on GitHub
- Review the code comments for technical details

## Acknowledgments

- India Meteorological Department for radar data
- Python scientific computing community
- Contributors to wradlib, xarray, and other libraries

---

**Quick Start Summary:**
1. `git clone https://github.com/Adityasinghvats/weather-radar-processing.git`
2. `python -m venv .venv`
3. `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
4. `pip install -r requirements.txt`
5. Place VOL files in project directory
6. `python voltonc.py`
7. `python processnc.py`
8. Check output files in CSV/JSON format
