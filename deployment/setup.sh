resourceGroup=undp-sids-rg
location=eastus
acrName=mlbackendacr
storageAccount=mlbackendsa
functionAppName=mlbackend
fullAcrName=$acrName.azurecr.io
imageName=$fullAcrName/core:latest
aksName=mlbackendCluster
aksShareName=aksshare

echo $resourceGroup, $location, $acrName, $storageAccount, $functionAppName, $fullAcrName, $imageName, $aksName

echo "Creating resource group $resourceGroup"
az group create --name $resourceGroup --location $location

echo "Creating container repo  $acrName"
az acr create --resource-group $resourceGroup --name $acrName --sku Basic

echo "Login to ACR with user $acrName"
az acr login --name $acrName
az acr update -n $acrName --admin-enabled true
password=$(az acr credential show -n $acrName --query 'passwords[0].value' -o tsv)
echo $password

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

# create a secret for the storage account
echo "Creating secrets"
storageConnectionString=$(az storage account show-connection-string --resource-group $resourceGroup --name $storageAccount --query connectionString --output tsv)
kubectl create secret generic storageaccountconnectionstring --from-literal=storageAccountConnectionString=$storageConnectionString
# create a secret for the acr credentials
kubectl create secret docker-registry containerregistrysecret --docker-server=$fullAcrName --docker-username=$acrName --docker-password=$password
az storage share create -n $aksShareName --connection-string $storageConnectionString
storageKey=$(az storage account keys list --resource-group $resourceGroup --account-name $storageAccount --query "[0].value" -o tsv)

kubectl create secret generic azure-secret --from-literal=azurestorageaccountname=$storageAccount --from-literal=azurestorageaccountkey=$storageKey
# attach the aks to the acr so it can grab the image from the acr
az aks update --name $aksName --resource-group $resourceGroup --attach-acr $acrName

echo "Generating kubernetes manifest file"
func kubernetes deploy --name $functionAppName --registry $fullAcrName --dry-run > ./deployment/k8_keda.yml

echo "Building docker image"
docker build -t $imageName .

echo "Pushing docker image"
docker push $imageName

echo "Do the required ./deployment/k8_keda.yml file and execute following command"
echo "kubectl apply -f ./deployment/k8_keda.yml"