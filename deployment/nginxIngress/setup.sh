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
subscriptionId=006c8a06-bc98-4f2e-a166-b56f87c77268
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
func kubernetes deploy --name $functionAppName --registry $fullAcrName  --namespace $namespace --dry-run > ./deployment/nginxIngress/k8_keda_main.yml

echo "Use following values for CI/CD. You must protect following credentials."
echo "REGISTRY_USERNAME"
echo $acrName

echo "REGISTRY_PASSWORD"
echo $password

echo "AZURE_CREDENTIALS"
az ad sp create-for-rbac --name $functionAppName --role contributor \
                                --scopes /subscriptions/$subscriptionId/resourceGroups/$resourceGroup \
                                --sdk-auth


#az network application-gateway ssl-cert create \
#   --resource-group $(az aks show --name $aksName --resource-group $resourceGroup --query nodeResourceGroup | tr -d '"') \
#   --gateway-name $aksGateway\
#   --name httpCert \
#   --cert-file certificate.pfx \
#   --cert-password <password>

#SOURCE_REGISTRY=k8s.gcr.io
#CONTROLLER_IMAGE=ingress-nginx/controller
#CONTROLLER_TAG=v1.0.4
#PATCH_IMAGE=ingress-nginx/kube-webhook-certgen
#PATCH_TAG=v1.1.1
#DEFAULTBACKEND_IMAGE=defaultbackend-amd64
#DEFAULTBACKEND_TAG=1.5
#CERT_MANAGER_REGISTRY=quay.io
#CERT_MANAGER_TAG=v1.5.4
#CERT_MANAGER_IMAGE_CONTROLLER=jetstack/cert-manager-controller
#CERT_MANAGER_IMAGE_WEBHOOK=jetstack/cert-manager-webhook
#CERT_MANAGER_IMAGE_CAINJECTOR=jetstack/cert-manager-cainjector
#
#az acr import --name $acrName --source $SOURCE_REGISTRY/$CONTROLLER_IMAGE:$CONTROLLER_TAG --image $CONTROLLER_IMAGE:$CONTROLLER_TAG
#az acr import --name $acrName --source $SOURCE_REGISTRY/$PATCH_IMAGE:$PATCH_TAG --image $PATCH_IMAGE:$PATCH_TAG
#az acr import --name $acrName --source $SOURCE_REGISTRY/$DEFAULTBACKEND_IMAGE:$DEFAULTBACKEND_TAG --image $DEFAULTBACKEND_IMAGE:$DEFAULTBACKEND_TAG
#az acr import --name $acrName --source $CERT_MANAGER_REGISTRY/$CERT_MANAGER_IMAGE_CONTROLLER:$CERT_MANAGER_TAG --image $CERT_MANAGER_IMAGE_CONTROLLER:$CERT_MANAGER_TAG
#az acr import --name $acrName --source $CERT_MANAGER_REGISTRY/$CERT_MANAGER_IMAGE_WEBHOOK:$CERT_MANAGER_TAG --image $CERT_MANAGER_IMAGE_WEBHOOK:$CERT_MANAGER_TAG
#az acr import --name $acrName --source $CERT_MANAGER_REGISTRY/$CERT_MANAGER_IMAGE_CAINJECTOR:$CERT_MANAGER_TAG --image $CERT_MANAGER_IMAGE_CAINJECTOR:$CERT_MANAGER_TAG
#
## Add the ingress-nginx repository
#helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
#
#helm install nginx-ingress ingress-nginx/ingress-nginx \
#    --version 4.0.13 \
#    --namespace default \
#    --set controller.replicaCount=2 \
#    --set controller.nodeSelector."kubernetes\.io/os"=linux \
#    --set controller.image.registry=$fullAcrName \
#    --set controller.image.image=$CONTROLLER_IMAGE \
#    --set controller.image.tag=$CONTROLLER_TAG \
#    --set controller.image.digest="" \
#    --set controller.admissionWebhooks.patch.nodeSelector."kubernetes\.io/os"=linux \
#    --set controller.admissionWebhooks.patch.image.registry=$fullAcrName \
#    --set controller.admissionWebhooks.patch.image.image=$PATCH_IMAGE \
#    --set controller.admissionWebhooks.patch.image.tag=$PATCH_TAG \
#    --set controller.admissionWebhooks.patch.image.digest="" \
#    --set defaultBackend.nodeSelector."kubernetes\.io/os"=linux \
#    --set defaultBackend.image.registry=$fullAcrName \
#    --set defaultBackend.image.image=$DEFAULTBACKEND_IMAGE \
#    --set defaultBackend.image.tag=$DEFAULTBACKEND_TAG \
#    --set defaultBackend.image.digest=""
#
## Public IP address of your ingress controller
#IP="<your external ip>"
#
## Name to associate with public IP address
#DNSNAME="ml-aks-ingress"
#
## Get the resource-id of the public ip
#PUBLICIPID=$(az network public-ip list --query "[?ipAddress!=null]|[?contains(ipAddress, '$IP')].[id]" --output tsv)
#
## Update public ip address with DNS name
#az network public-ip update --ids $PUBLICIPID --dns-name $DNSNAME
#
## Display the FQDN
#az network public-ip show --ids $PUBLICIPID --query "[dnsSettings.fqdn]" --output tsv
#
#
## Label the ingress-basic namespace to disable resource validation
#kubectl label namespace ingress-basic cert-manager.io/disable-validation=true
#
## Add the Jetstack Helm repository
#helm repo add jetstack https://charts.jetstack.io
#
## Update your local Helm chart repository cache
#helm repo update
#
## Install the cert-manager Helm chart
#helm install cert-manager jetstack/cert-manager \
#  --namespace default \
#  --version $CERT_MANAGER_TAG \
#  --set installCRDs=true \
#  --set nodeSelector."kubernetes\.io/os"=linux \
#  --set image.repository=$fullAcrName/$CERT_MANAGER_IMAGE_CONTROLLER \
#  --set image.tag=$CERT_MANAGER_TAG \
#  --set webhook.image.repository=$fullAcrName/$CERT_MANAGER_IMAGE_WEBHOOK \
#  --set webhook.image.tag=$CERT_MANAGER_TAG \
#  --set cainjector.image.repository=$fullAcrName/$CERT_MANAGER_IMAGE_CAINJECTOR \
#  --set cainjector.image.tag=$CERT_MANAGER_TAG