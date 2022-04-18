# Define variables
resourceGroup=ml-backend-group
location=eastus
acrName=acrmlbackend
storageAccount=mlbackendstorage
functionAppName=sidmlbackend
fullAcrName=$acrName.azurecr.io
imageName=$fullAcrName/$functionAppName:latest
aksName=sidMLBackendCluster
aksShareName=aksshare
subscriptionId=<subscription Id>
aksAppGateway=mlbackendAG
namespace=ml-app

echo "Creating resource group $resourceGroup"
az group create --name $resourceGroup --location $location

echo "Creating container repo  $acrName"
az acr create --resource-group $resourceGroup --name $acrName --sku Basic

echo "Login to ACR with user $acrName"
az acr login --name $acrName
az acr update -n $acrName --admin-enabled true
password=$(az acr credential show -n $acrName --query 'passwords[0].value' -o tsv)

echo "Creating storage $storageAccount"
az storage account create --name $storageAccount --location $location --resource-group $resourceGroup --sku Standard_LRS

echo "Creating AKS cluster"
az aks create \
  --resource-group $resourceGroup \
  --name $aksName \
  --node-count 1 \
  --vm-set-type VirtualMachineScaleSets \
  --load-balancer-sku standard \
  --enable-cluster-autoscaler \
  --min-count 1 \
  --max-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

az aks get-credentials --resource-group $resourceGroup --name $aksName

echo "Installing aks cli"
az aks install-cli

kubectl create namespace $namespace
# create a secret for the storage account
echo "Creating secrets"
storageConnectionString=$(az storage account show-connection-string --resource-group $resourceGroup --name $storageAccount --query connectionString --output tsv)
kubectl create secret generic storageaccountconnectionstring --namespace=$namespace --from-literal=storageAccountConnectionString=$storageConnectionString
# create a secret for the acr credentials
kubectl create secret docker-registry containerregistrysecret --namespace=$namespace --docker-server=$fullAcrName --docker-username=$acrName --docker-password=$password
az storage share create -n $aksShareName --connection-string $storageConnectionString
storageKey=$(az storage account keys list --resource-group $resourceGroup --account-name $storageAccount --query "[0].value" -o tsv)

kubectl create secret generic azure-secret --namespace=$namespace --from-literal=azurestorageaccountname=$storageAccount --from-literal=azurestorageaccountkey=$storageKey
# attach the aks to the acr so it can grab the image from the acr
az aks update --name $aksName --resource-group $resourceGroup --attach-acr $acrName

echo "Building docker image"
docker build -t $imageName .

echo "Pushing docker image"
az acr login -n $acrName
docker push $imageName

echo "Generating kubernetes manifest file"
func kubernetes deploy --name $functionAppName --registry $fullAcrName  --namespace $namespace --dry-run > ./deployment/nginxIngress/manifests/k8_keda_main.yml

echo "Use following values for CI/CD. You must protect following credentials."
echo "REGISTRY_USERNAME"
echo $acrName

echo "REGISTRY_PASSWORD"
echo $password

echo "AZURE_CREDENTIALS"
az ad sp create-for-rbac --name $functionAppName --role contributor \
                                --scopes /subscriptions/$subscriptionId/resourceGroups/$resourceGroup \
                                --sdk-auth


