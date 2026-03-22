import numpy as np
import pandas as pd

# Assuming forecast_df and filtered DataFrames are generated in the code prior to this.

# Update for plotting only future predictions.
plot_history = filtered[["date","sales"]].copy()
plot_history["prediction"] = np.nan
plot_data = pd.concat([plot_history, forecast_df], ignore_index=True)

# Keep in-sample predictions for MAE calculation.
