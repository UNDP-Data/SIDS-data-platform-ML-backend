# SID ML Backend

## Introduction
Implementation for the SID ML model execution backend. Currently, implemented only for K-NN Imputes based prediction 
model as exposed it as an azure function service inside a Kubernetes cluster. 

## System Architecture

<img src="./docs/images/layout.png?raw=true" height="500px">

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
    
    These values can update from [k8_keda.yml](./deployment/k8_keda.yml) file


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

## Deployment

### Setup Azure Kubernetes Cluster
1. Install Azure core tools - [link](https://github.com/Azure/azure-functions-core-tools).
2. Install Azure CLI - [link](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli)
3. Install docker
4. Update variables in [setup.sh](./deployment/setup.sh) based on the requirement.
5. Run ```./deployment/setup.sh``` script file from the project root directory.
6. At the end, it will print three variables that we need for Github Action CI/CD. Save them in a safe place. Do not share!
7. Above command will create all the resources required for the cluster and create a Kubernetes configuration file in the deployment 
folder as k8_keda.yml. Update the file based on the requirement.
8. Run ```kubectl apply -f ./deplyment/k8_keda.yml``` to create Kubernetes services and deployments in the cluster.
9. Cluster will be ready in a few minutes. Can access the cluster from Azure Console Kubernetes Services -> Services and Ingress -> External IP
10. View swagger documentation from http://<cluster ip>/docs

### CI/CD
CI/CD implemented using Github Actions.[config file](./.github/workflows/main.yml). It performs the following actions
- Build a docker container and push it to the Azure Container Registry (ACR)
- Perform rollout pod restart in Kubernetes cluster.

### Steps to Add New Model API
1. Create new python module inside models similar to `imputation`
2. Create a `__init__.py` file inside newly created module and define all the fast api endpoints. 
   Example
    ```
    from typing import Optional
    from fastapi import APIRouter
    from pydantic import BaseModel, Field
    
    
    class SampleRequest(BaseModel):
        requiredField: str = Field(None, title="This field is required", example="required 123")
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
This endpoint will automatically add in to the swagger documentation as a new session. 

**IMPORTANT: `router` object name must be same in every model. All other functions and variables can rename and rearrange the way you want**

For readability message definitions can move in to a separate file. 

Add custom validators for message fields at the API level as below. 
```
@validator('requiredField')
def username_alphanumeric(cls, v):
    assert v.isalnum(), 'must be alphanumeric'
    return v
 ```
please refer [pydantic validators](https://pydantic-docs.helpmanual.io/usage/validators/) for more information.

3. Use `DATASET_PATH` environment variable for dataset loading. 

## Testing

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
http://20.88.191.216/docs#/

Kubernetes endpoint is faster than consumer plan endpoint.
