# Compute history_series and forecast_series as before
history_series = ...
forecast_series = ...

# Deriving last_actual_date/value when available
if history_series:
    last_actual_date = history_series.index[-1]
    last_actual_value = history_series.iloc[-1]
else:
    last_actual_date = None
    last_actual_value = None

# Build forecast_plot by prepending the last actual point to forecast_series
if last_actual_date and last_actual_value is not None:
    forecast_plot = [last_actual_value] + forecast_series.tolist()
else:
    forecast_plot = forecast_series.tolist()

# Using existing Plotly graph_objects figure and layout
fig.add_trace(go.Scatter(x=history_series.index, y=history_series, mode='lines', name='Actual Sales'))
fig.add_trace(go.Scatter(x=forecast_series.index, y=forecast_plot, mode='lines', name='Forecasted Sales'))