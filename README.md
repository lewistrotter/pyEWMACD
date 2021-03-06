# pyEWMACD
## Summary
This is an early (but working) python implementation of Brooks et al., (2017) EWMACD (including EDYN). It is a almost 1:1 conversion of Brooks' R code (version 1.9.6) located here: https://vtechworks.lib.vt.edu/handle/10919/50543.5. We have added support for Xarray Datasets (thus NetCDF) as input into the EWMACD function to simplify Brooks' implemntation. All you need is an xarray time series dataset with x, y, time (numpy datetime64) coordinates and a variable of vegetation values and you are good to go. See the example below.

This script is still very rough (had to get it done quick for a project), but I will clean up and improve in the future. We'd like to thank Brooks et al. for their great work on EWMACD. If you use this script, please cite the reference at the bottom of this page if you publish your research.

## EWMACD and EDYN
Based on Brooks et al. (2017), Exponentially Weighted Moving Average Change Detection (EWMACD) is a freely available, open-source pixel-level time series change detection algorithm originally designed to detect a wide variety of persistent changes to forested pixels. EWMACD uses exponentially weighted moving average (EWMA) control charts to analyze residual values resulting from fitting the input time series to harmonic (e.g., Fourier) curves to account for seasonal patterns.

The result is a time series of signals which convey not only the presence of a disturbance but also the magnitude and timing, up to the temporal resolution of the input data. Part of the class of memory control charts, EWMA charts are specifically designed to detect subtle shifts from the in-control state, the state in which a process (in this case, forest status) continues to behave according to its historically observed or intended characteristics (e.g., stable forest). This makes them ideally suited to detect not only acute changes, such as harvests and fires, but also longer, slower periods of gradual forest decline.

This script offers two of Brooks' EWMACD impmentation; 'static' and 'dynamic' (i.e., EDYN). The former continously uses the same training period set by the user to derive residuals, whereas the latter will automatically reset the training period to a new 'regime' once a significant break has occurred. See Brooks et al., (2017) for more information.

## Demonstration
```python
# pyEWMACD only requires a few common libraries
import os
import sys
import time
import numpy as np
import xarray as xr
import scipy.stats
```

```python
# take xarray dataset with x, y, time coordinates a variable called veg_idx containing raw ndvi values
ds = xr.Dataset(...)
```

```python
# feed the dataset into the EWMACD function
out = EWMACD(ds=ds, 
             trainingPeriod='static',
             trainingStart=2000,
             trainingEnd=2005,
             persistence_per_year=1)
```

```python
# plot the output variable called 'change' (on a per-pixel basis)
import matplotlib.pyplot as plt
out['change'].isel(x=0, y=0).plot()
plt.show()
```

# References
Brooks, E.B., Yang, Z., Thomas, V.A. and Wynne, R.H., 2017. Edyn: Dynamic signaling of changes to forests using exponentially weighted moving average charts. Forests, 8(9), p.304.
