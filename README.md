This is a repository from ML learning backend

### Run Locally
1. Install Azure core tools - https://github.com/Azure/azure-functions-core-tools.
2. Copy dataset files to dataset folder in root directory.
3. Run `func start host`. This will start the azure function locally.
4. View swagger API from http://localhost:7071/docs.

### Test requests
Simple faster request
```json
{
  "manual_predictors": ["wdi-AG.LND.AGRI.K2"],
  "target_year": "2001",
  "target": "key-wdi-EG.ELC.ACCS.ZS",
  "interpolator": "KNNImputer",
  "scheme": "Manual",
  "estimators": 10,
  "model": "rfr",
  "interval": "100"
}
```


Time consuming request
```json
{
  "number_predictor": 10,
  "target_year": "2001",
  "target": "key-wdi-EG.ELC.ACCS.ZS",
  "interpolator": "KNNImputer",
  "scheme": "Automatic via feature selection",
  "estimators": 100,
  "model": "rfr",
  "interval": "100"
}
```

### Azure function - Swagger Documentation (consumer plan)
https://sidsapi-basic.azurewebsites.net/docs#/

### Kubernetes - Swagger Documentation
http://20.88.191.216/docs#/

Kubernetes endpoint is faster that consumer plan endpoint