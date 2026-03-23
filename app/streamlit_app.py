# Update for Sales Forecast View
# New lines added to handle history and anchor value

# Assuming the necessary imports and initializations are present above

# Set BUILD_VERSION
BUILD_VERSION = '2026-03-22-1738'

# Sales Forecast View
# Convert sales/prediction to numeric
# ... (existing code) ...

nonzero_history = history_series.loc[history_series["sales"].notna() & (history_series["sales"] > 0)]
anchor_date = last_actual_date
anchor_value = last_actual_value
if anchor_value is None and not nonzero_history.empty:
    anchor_value = float(nonzero_history["sales"].iloc[-1])
if anchor_value == 0 and not nonzero_history.empty:
    anchor_value = float(nonzero_history["sales"].iloc[-1])

# Forecast plot block
forecast_offset = anchor_value  # replace last_actual_value with anchor_value
# ... (rest of forecast plot code) ...

# Forecast Debug output
print(f"Anchor Value: {anchor_value}, Anchor Date: {anchor_date}")

# Keep everything else unchanged
# ... (remaining code) ...