import pandas as pd
import numpy as np
from smartmoneyconcepts import smc

# Create a dummy dataframe
data = {
    "open": [10, 11, 12, 11, 10, 9, 8, 9, 10, 11] * 10,
    "high": [11, 12, 13, 12, 11, 10, 9, 10, 11, 12] * 10,
    "low": [9, 10, 11, 10, 9, 8, 7, 8, 9, 10] * 10,
    "close": [10.5, 11.5, 11.5, 10.5, 9.5, 8.5, 8.5, 9.5, 10.5, 11.5] * 10,
    "volume": [100] * 100
}
df = pd.DataFrame(data)

# Calculate swings and OB
swings = smc.swing_highs_lows(df, swing_length=5)
ob_data = smc.ob(df, swing_highs_lows=swings)

print("OB Data Columns:", ob_data.columns)
print("First 20 rows of OB Data:")
print(ob_data.head(20))

print("\nValue counts for OB column:")
print(ob_data["OB"].value_counts())

# Filter for start of OB
is_new_ob = (ob_data["OB"] != 0) & (
    (ob_data["OB"] != ob_data["OB"].shift(1)) | 
    (ob_data["Top"] != ob_data["Top"].shift(1))
)
print("\nNumber of new OBs detected with filter:", is_new_ob.sum())
