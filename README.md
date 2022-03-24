This is a repository from ML learning backend

### Run Locally
1. Install Azure core tools - https://github.com/Azure/azure-functions-core-tools.
2. Copy dataset files to dataset folder in root directory.
3. Run `func start host`. This will start the azure function locally.
4. View swagger API from http://localhost:7071/docs.

### Test request
`
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
`

### Docker Image Build
docker build --tag palinda/sidsbackend:v1.0.0 .      

### Docker Image Push
docker push palinda/sidsbackend:v1.0.0

This will automatically deploy the changes to the Azure function

### System Swagger Documentation
https://sidsapi.azurewebsites.net/docs#/