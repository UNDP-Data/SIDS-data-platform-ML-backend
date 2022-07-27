# SID ML Backend

## Introduction
Implementation for the SID ML model execution backend. Currently, implemented only for K-NN Imputes based prediction 
model as exposed it as an azure function service inside a Kubernetes cluster. 

## System Architecture

<img src="./docs/images/layout2.png?raw=true" height="500px">

The system layout on Azure Kubernetes Service(AKS) is shown in the above diagram. Kubernetes cluster created inside an Azure Virtual Private Network (VNET), and all the external traffic coming through
Azure load balancer resource configures an external IP address and connects the requested pods to the load balancer backend pool. Load balancing rules are created on the desired ports to allow customers' traffic to reach the application.

The machine learning model was developed in python and exposed as a RestAPI using FastAPI python framework wrapped inside an Azure function.
Since it is wrapped as an Azure function, we can deploy it into the Azure cloud in multiple ways. 
1. Azure Function Service - Serverless Platform (Consumer plan)
2. Azure Kubernetes Service(AKS) - Azure function package in a Docker container and deploy. 

In the AKS context, Currently, we have only a single app service, and it will serve all the client requests. The ML-Backend is developed 
in the way that can support multiple models as a single Kubernetes service or multiple Kubernetes services. 
- Single Service is suitable if we use the same dataset for different models or multiple small datasets.
- Multiple Services is suitable when we have multiple large datasets. With this approach, we can maximize node utilization. If we need to add a new service, we need to update the Kubernetes manifest and update the cluster.
  
### Scalability
Current Kubernetes deployment automatically scales up and down using two ways,
1. **Cluster Autoscaler** - Watches for pods that can't be scheduled on nodes because of resource constraints. 
   The cluster then automatically increases the number of nodes. The current cluster is configured for a maximum of three nodes and a minimum of one node.
   Use the following Azure CLI command to update the autoscaler configuration.
   ```
   az aks update --resource-group myResourceGroup --name myAKSCluster --update-cluster-autoscaler --min-count 1 --max-count 5
   ``` 
   
2. **Horizontal pod autoscaler** - Uses the Metrics Server in a Kubernetes cluster to monitor the resource demand of pods. 
   If an application needs more resources, the number of pods is automatically increased to meet the demand.
   Current parameters
   - Minimum Replicas: 1
   - Maximum Replicas: 5
   - Target scale CPU utilization percentage: 50
    
    These values can update from [k8_keda.yml](deployment/nginxIngress/k8_keda.yml) file


   ![Scaling Options](./docs/images/cluster-autoscaler.png?raw=true "Title")

Please refer [link](https://docs.microsoft.com/en-us/azure/aks/cluster-autoscaler) for more information.


### Security
Since the Kubernetes cluster is created inside a private network, an external party does not have direct access to the cluster 
except through the load balancer. 

API CORS enabled only for following domains
   - https://lenseg.github.io/SIDSDataPlatform/
   - https://sids-dashboard.github.io/SIDSDataPlatform/
   - http://localhost
   - http://localhost:8080

API support only GET, POST and OPTIONS HTTP methods. 

Currently, I did not enable any of the extra security features provided by Azure since they have a cost and can configure based on the requirement.
Security features supported by Azure
- DDoS Protection - [Pricing](https://azure.microsoft.com/en-gb/pricing/details/ddos-protection/)
- Firewall - [Pricing](https://azure.microsoft.com/en-us/pricing/details/azure-firewall/#pricing)


### Fault Tolerance
Since the current minimum replica count and minimum node count is one, there can be 2~3 minutes of system unavailability
. If we increase it to above 1, it will minimize the unavailable probability.
But the system automatically receivers from any system unavailability.

### Storage
Using Azure Storage File Share to store model datasets. File Share mounted on the docker container at the startup as a volume with read and write access. 
All the pods sharing the same storage. 

## Developer Guide
### Folder Structure
Source code repository structured in the following way

- **main** - Root RESTAPI implementation and azure function configuration
- **api_app** - FastAPI application configuration
- **shared_dataloader** - Shared data loader to share data across multiple models
- **common**: 
   - utility.py - System wide utility functions 
   - constants.py - System wide Constants 
   - errors.py - Define all the RestAPI custom error messages
- **deployment**:
   - setup.sh - Initial Kubernetes cluster related resource creation and configuration
   - k8_keda.yml - Kubernetes replaces configuration. Can use this file to update cluster configuration. 
     Currently, changes to this file do not automatically deploy into the system via Github Actions. Need to 
     manually run ```kubectl apply -f ./deployment/k8_keda.yml```
- **models** - Contains individual model implementation, RestAPI endpoints, model-specific constants and message definitions. 

## Available models
- **twolvlImp** : This is an imputation model that for filling missing data for SIDS in country level indicator. In this approach (year-by-year approach) only a single year of data is taken into account for imputation. After which a target column is imputed according to the following methodology. 
  - In the pre-processing stage, the column of the target year is sliced from the original dataset. Then using pandas’ stack function the sliced data frame is reshaped into a country by indicator format. 
  - The new reshaped data frame is split into training and testing data based on the target column (indicator to be imputed). 
  - All columns, except the target column, are ranked by the amount of missing values they contain in the training data frame. For a given column, if more than 40% of the values are missing (i.e., measure = 40) the column will not be considered as a possible predictor and is removed from both training and testing data frames. 
  - Then, missing data in the predictor column is imputed using a standard imputer. The current implementation [KNN imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html), [simple imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html#sklearn.impute.SimpleImputer) and [iterative imputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html#sklearn.impute.IterativeImputer). 
  - After which dimensionality reduction is performed on the imputed predictors of the training data frame, reducing their number to 10. Available options include Recursive feature selection, principal component analysis and manual selection of features based on prior knowledge. PCA dimensionality reduction is the least computational intensive method and hence, used for generating current results. The selected features are also used to subset the test data frame.
  - Then the model trainer function is used to fit the selected (or best) model and generate [prediction intervals](https://saattrupdan.github.io/2020-03-01-bootstrap-prediction/). Cross-validation is used to explore the parameter space via gridsearch and measure generalizability via normalized root mean squared error. The models available include, 'Random Forest Regressor', 'Gradient Boost Regressor','Extra tree Regressor'.  Due to the small size of data, both SIDS and non-SIDS are used to train the selected model. In order the accommodate for the small sample size of SIDS, SIDS observation are given higher weight during training using Inverse class weighting.
- **CountryCorrelation** : contain functions for computing K-Means clustering and correlation on data with missing values.
  - Correlation Function: Computes and returns countries most closely correlated to the selected SIDS country based on a subset of indicators (sub selected by category) for a given year. The original data is sliced based on category and year then reshaped using pandas’ stack functionality into a country by indicator format.
  - Cluster Function: Clusters SIDS countries based on k-means algorithm. The original data is sliced based on category of indicators and year then reshaped using pandas’ stack functionality into a country by indicator format. Then k-means in combination with EM algorithm is used to fill gaps and assign countries to k clusters.
- **Statimpute** : This folder contains the imputer function for interpolating an indicator's value across time for a given SIDS country. The function makes use of pandas [interpolate](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) method and provides several methods such as linear, quadraitc, spline, etc.
- **SurveyCluster** : This folder contains models for clustering crisis brueau survey data based on an accompanying metadata strcuture. K-modes clustering for categorical features, K-means clustering for numeric features and K-prototypes clustering for mixed group of features. The elbow method is also implemented to choose an optimal number of clusters.
 

## Deployment

### Setup Azure Kubernetes Cluster

#### Prerequisites
1. Azure core tools - [link](https://github.com/Azure/azure-functions-core-tools).
2. Azure CLI - [link](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
3. Terraform - [link](https://learn.hashicorp.com/tutorials/terraform/install-cli)
   
#### New AKS Cluster Creation
1. Update variables in [main.tf](./deployment/terraform/main.tf). Variable descriptions mentioned in the file [variables.tf](./deployment/terraform/main/variables.tf).
2. Login to Azure cli by executing ``az login``
2. Run ``terraform apply`` from the project [deployment/terraform](./deployment/terraform) directory. This will create all the resources and kubernetes yml files. (If this command fail in the first attempt retry another time.)
3. Upload the dataset files in to Azure file storage datasets container.
4. Now we can set up CI/CD from GitHub actions. For that you need to add following secrets to GitHub repo,
    1. ``REGISTRY_USERNAME = <container registry name>``
    2. ``REGISTRY_PASSWORD = <container registry password``. Get the password for Azure Container registry by executing command ``az acr credential show -n <registry name> --query 'passwords[0].value' -o tsv``. 
    3. ``AZURE_CREDENTIALS = <credentials json>``. To get credentials execute command ``az ad sp create-for-rbac --name <app name> --role contributor \
                                --scopes /subscriptions/<subscription id>/resourceGroups/<resource group name> \
                                --sdk-auth``
       
5. Update [main.yml](./.github/workflows/main.yml) env variables based on your values
6. Commit all the changes to the git repo. It will trigger GitHub workflow and deploy all the Kubernetes services to the AKS cluster.
7. Cluster will be ready in a few minutes. Can get the cluster public ip from Azure Console Kubernetes Services -> Services and Ingresses -> Ingress -> External Address
8. View swagger documentation from ``http://<public ip>/docs``

#### Deploy on Existing Cluster
Import your existing resources using ``terraform import <module.<module_name>.<resource module name>.<resource name> /subscriptions/<subscription id>/<resource address>``. 
   E.g.,: Importing existing resource group named `ml-backend-group`
   ```
   terraform import module.dev_cluster.azurerm_resource_group.rg \
  /subscriptions/<subscription id>/resourceGroups/ml-backend-group
   ``` 
#### Add Different Node Pool for Specific Service
1. Create a new node pool with different configuration by adding below code to the Terraform cluster_creation.tf file. Change values based on your requirement.
```
resource "azurerm_kubernetes_cluster_node_pool" "survey" {
  name                  = "survey"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.clusterNGINX[0].id
  vm_size               = "Standard_DS2_v2"
  node_count            = 1

  tags = {
    Environment = "Testing"
  }
}
```
2. Then we can set specific kubernetes deployment for this node by adding following code under deployment spec. [More](https://kubernetes.io/docs/tasks/configure-pod-container/assign-pods-nodes/)
```
nodeSelector:
    agentpool: survey
```

### CI/CD
CI/CD implemented using Github Actions. [config file](./.github/workflows/main.yml). It performs the following actions
- Build a docker container and push it to the Azure Container Registry (ACR)
- Perform rollout pod restart in Kubernetes cluster.

### Steps to Add New Model API
1. Create new python module inside models similar to `sampleapi`
2. Create a `__init__.py` file inside newly created module and define all the fast api endpoints. `tags` should be the description for the model.
   Example
    ```
    from typing import Optional
    from fastapi import APIRouter
    from pydantic import BaseModel, Field
    
    
    class SampleRequest(BaseModel):
        requiredField: str = Field(..., title="This field is required", example="required 123")
        optionalField: Optional[str] = Field(None, title="This field is optional", example="optional 123")
    
    
    class SampleResponse(BaseModel):
        resp1: str
    
    
    router = APIRouter(
        prefix="/sample_model",
        tags=["Sample Model"],
        responses={404: {"description": "Not found"}},
    )
    
    
    @router.post('/test_endpoint1', response_model=SampleResponse)
    async def test_endpoint1(req: SampleRequest):
        return SampleResponse(resp1="Test 1")
    
    
    @router.post('/test_endpoint2')
    async def test_endpoint1(name: str):
        return "Hi "+name

   ```
    This endpoint will automatically add in to the swagger documentation as a new session.<br><br>

    **IMPORTANT: Every model must have APIRouter object named as `router`.**
    Reason for the above compulsory change:
    - `router` object used for the FastAPI app routing. 
      If you don't have a variable like this, your model will not visible on the swagger api and REST API endpoints 
      will not available. All other functions and variables can rename and rearrange the way you want.
    
3.  Root scope `/params` endpoint returns the all models main endpoint definition. For that you need to add `openapi_extra={MAIN_ENDPOINT_TAG: True}` to the endpoint.
    E.g.
    ```
    @router.post('/test_endpoint1', response_model=SampleResponse, openapi_extra={MAIN_ENDPOINT_TAG: True})
    async def test_endpoint1(req: SampleRequest):
        return SampleResponse(resp1="Test 1")
    ```

4. Message definitions should be derived from `BaseDefinition`. It will  add `required_if` basic validation support to the definition.
    For readability message definitions can move in to a separate file. 

    Add custom validators for message fields at the API level as below. 
    ```
    @validator('requiredField')
    def username_alphanumeric(cls, v):
        assert v.isalnum(), 'must be alphanumeric'
        return v
     ```
    please refer [pydantic validators](https://pydantic-docs.helpmanual.io/usage/validators/) for more information.


5. Use `DATASET_PATH` environment variable for dataset loading.
6. If you are shared same data loading across multiple models, create a data loader in `shared_dataloader` similar to [indicator_dataloader.py](./shared_dataloader/indicator_dataloader.py) and loaded data in the model as below code
   ```
   indicatorMeta, datasetMeta, indicatorData = data_loader.load_data("<model service name>", data_importer())
   ```
7. By default, newly added endpoint will route through default Kubernetes service.

#### Add Model Endpoint as a New Kubernetes Service
1. It is better to serve as a different service on following reasons
   - Different datasets. 
   - Different resource usage.
2. It will enable independent resource configurability and scalability.
3. Deployment steps as follows,
   1. Execute command `python deployment/service_gen.py`. Script will request following information. 
      - **Service name** : This will be the Kubernetes service name. Avoid duplicate service names.
      - **Model folder name** : This will be the folder name that added to the `models` python module.
      - **Shared Volume** : If you need to have a storage that contains sharable resources across all the pods (ex: datasets). Selected `y` else `n`
      - **Type of shared volume** :
         - file - Azure Storage file share
         - blob - Azure Storage blob storage
      - **Name of the shared volume**
         - file - File share name
         - blob - Container name
      - **Requested memory** : Amount of memory allocated at the pod startup. [Memory units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory)
      - **Requested cpu** : Amount of cpu allocated at the pod startup. [CPU units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#resource-units-in-kubernetes)
      - **Memory limit** : Maximum amount of memory allocated for the pod. [Memory units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#meaning-of-memory)
      - **CPU limit** : Maximum amount of cpu allocated for the pod. [CPU units](https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#resource-units-in-kubernetes)
      
   2. When you develop python model, if you are doing model specific resource loading at the startup, please check the `SERVICE_MODEL` env variable to avoid unnecessary resource usage.
   3. Once you commit these new changes, Cluster will be automatically update via CI/CD.

## Testing

### Load Testing
#### Cluster Configuration
Tested using Jmeter tool.

*Node Configuration*

|Size|vCPU|Memory: GiB|Expected network bandwidth (Mbps)|
| :---: | :---: |:---: |:---: |
|Azure Standard_DS2_v2|2|7|1500

- Minimum Node Count : 1
- Maximum Node Count : 5


*Pod Configuration*
   - Pod Resource Limit
      - CPU: 1
      - Memory: 2 GiB
   
   - Pod Resource Request
      - CPU: 0.5
      - Memory: 1 GiB


- Minimum Pod Replicas : 1
- Maximum Pod Replicas : 6

#### Test Scenario
Sent `Simple Request` mentioned below from 20 users within 1 minute, and repeat it for 20 times.

#### Results
1. Cluster started with single node, single pod.
2. Within first 40 seconds expected pod count increased to 4 and node count increased to 3.
3. All extra pods and nodes started within next 1 minute. 
4. Within this period received set of Gateway-Timeout responses. 
5. In next 1 minute expected pod count increased to 6, node count increased to 4.
6. All pods up and running and responses got stable. 
7. Result as below

| Request Count | Error Rate | Throughput |
   | :---: | :---: | :---: |
|400|30%|34.6 /min|

<img src="./docs/images/ResponseTimeGraph.png?raw=true" height="400px">

8. 10 minutes after the test, pods and node count dropped to 1.

### Local Environment Setup
1. Install Azure core tools - [link](https://github.com/Azure/azure-functions-core-tools).
2. Copy dataset files to the dataset folder in the root directory.
3. Run `func start host`. This will start the azure function locally.
4. View swagger API from http://localhost:7071/docs.

Simple  request
```json
{
  "manual_predictors": ["wdi-AG.LND.AGRI.K2"],
  "target_year": "2001",
  "target": "key-wdi-EG.ELC.ACCS.ZS",
  "interpolator": "KNNImputer",
  "scheme": "MANUAL",
  "estimators": 10,
  "model": "rfr",
  "interval": "quantile"
}
```


Time-consuming request
```json
{
  "number_predictor": 10,
  "target_year": "2001",
  "target": "key-wdi-EG.ELC.ACCS.ZS",
  "interpolator": "KNNImputer",
  "scheme": "AFS",
  "estimators": 100,
  "model": "rfr",
  "interval": "quantile"
}
```
## Notes
### Deployments - Swagger Documentation
#### Azure function - consumer plan
https://sidsapi-basic.azurewebsites.net/docs#/

#### Kubernetes - AKS
https://ml-aks-ingress.eastus.cloudapp.azure.com/docs

Kubernetes endpoint is faster than consumer plan endpoint.

### Time Estimation Automation
Currently, added a log in the two level imputation model and got a moving time average by following query.
```
ContainerLog
| parse LogEntry with * "Time Consumed(s)=" Time " " other_params " scheme=<Schema." scheme_e ": '" scheme "'" remain_params
| where LogEntry contains "Time Consumed(s)="
| extend time_parsed    = toint(Time)
| summarize avg(time_parsed) by scheme
```
Automated the process using [Azure Logic App](./deployment/TimeConsumptionAutomationTaskTemplate.json). Logic app generating a [JSON file](./time_consumption) in `dataset` Azure Storage Container.