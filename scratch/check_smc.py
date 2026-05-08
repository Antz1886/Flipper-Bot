
import pandas as pd
import numpy as np
from smartmoneyconcepts import smc

# Create dummy OHLC data
data = {
    "open":  [100, 102, 105, 106, 110],
    "high":  [105, 107, 108, 112, 115],
    "low":   [98,  100, 104, 105, 108],
    "close": [102, 105, 106, 110, 112],
    "volume":[1000, 1100, 1200, 1300, 1400]
}
df = pd.DataFrame(data)

print("Calling smc.fvg...")
try:
    fvg = smc.fvg(df)
    print("FVG columns:", fvg.columns.tolist())
    print(fvg)
except Exception as e:
    print("Error in smc.fvg:", e)

print("\nCalling smc.swing_highs_lows...")
try:
    swings = smc.swing_highs_lows(df, swing_length=2)
    print("Swings columns:", swings.columns.tolist())
    
    print("\nCalling smc.ob...")
    ob = smc.ob(df, swing_highs_lows=swings)
    print("OB columns:", ob.columns.tolist())
    print(ob)
except Exception as e:
    print("Error in OB detection:", e)
