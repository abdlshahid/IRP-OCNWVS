import cdsapi
import os
from datetime import datetime
from calendar import monthrange

c = cdsapi.Client()

year = 2019

# Start from month 2 since month 1 is already downloaded
for month in range(1, 13):
    # Get first and last day of the month
    first_day = 1
    last_day = monthrange(year, month)[1]
    
    # Format date range for this month
    date_start = f"{year}-{month:02d}-{first_day:02d}"
    date_end = f"{year}-{month:02d}-{last_day:02d}"
    date_range = f"{date_start}/to/{date_end}"
    
    # Create output directory structure: data/year/month/
    output_dir = f"eval_data/gum/{year}/{month:02d}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Output filename: era5_wave_YYYYMM.nc
    output_file = f"{output_dir}/era5_wave_{year}{month:02d}.nc"
    
    print(f"Downloading data for {year}-{month:02d} ({date_range})...")
    print(f"Output: {output_file}")
    
    try:
        c.retrieve(
            'reanalysis-era5-complete',
            {
                'class': 'ea',
                'expver': '1',
                'stream': 'wave',
                'type': 'an',
                'param': '251.140',   # 2D directional wave spectra

                'date': date_range,
                'time': '00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/'
                        '06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/'
                        '12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/'
                        '18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00',

                'frequency': '/'.join(str(i) for i in range(1,31)),
                'direction': '/'.join(str(i) for i in range(1,25)),

                'area': [27.5, -90.5, 26.5, -89.5],  # [North, West, South, East]

                'grid': '0.5/0.5',
                'format': 'netcdf',
            },
            output_file)
        
        print(f"✓ Completed: {output_file}\n")
        
    except Exception as e:
        print(f"✗ Error downloading {year}-{month:02d}: {e}\n")
        continue

print("All downloads complete.")