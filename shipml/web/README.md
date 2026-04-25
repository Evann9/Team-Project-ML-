# Ship Route Flask Map

Run from the repository root:

```powershell
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\type_anal\apply_type_model_to_routes.py
C:\Users\green\anaconda3\envs\myproject\python.exe shipml\web\app.py
```

Then open http://127.0.0.1:5000.

The first command trains or loads the best available ship-type classifier, applies it
to `route_predictions.csv`, and writes `route_predictions_with_types.csv`. The Flask
app uses that enriched CSV when it exists, otherwise it falls back to the raw route
prediction CSV with ship type shown as `Unknown`.
