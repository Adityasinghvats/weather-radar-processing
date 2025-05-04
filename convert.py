import xarray as xr
import pandas as pd

ds = xr.open_dataset("2024062612321800dBZ.vol.nc", decode_times=False)
df = ds.to_dataframe().reset_index()
# df.to_csv('output.csv',index=False)
# find out any way to handle dbz.vol file
print(df['reflectivity'].dims)
print(df['reflectivity'].shape)