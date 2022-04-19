variable "resourceGrp" {
  description = "Azure AKS related resource group"
}

variable "location" {
  description = "Resource Location"
}

variable "acrName" {
  description = "Container repository name"
}

variable "storageAccountName" {
  description = "Azure Storage Account Name"
}

variable "aksClusterName" {
  description = "AKS Cluster name"
}

variable "sharedDataContainerName" {
  description = "Shared blob storage container name"
}

variable "env" {
  description = "Environment type"
}

variable "clusterNodeCount" {
  description = "AKS cluster initial node count"
}

variable "clusterMinNodeCount" {
  description = "AKS node auto scaler minimum node count"
}

variable "clusterMaxNodeCount" {
  description = "AKS node auto scaler maximum node count"
}

variable "clusterNodeSize" {
  description = "AKS cluster node size"
}

variable "appNamespace" {
  description = "AKS cluster ML backend namespace"
}

variable "appName" {
  description = "AKS main service name"
}

variable "createAzureAppGW" {
  description = "Create Azure Application gateway(true) or use default nginx ingress(false)"
}

variable "useBlobStorage" {
  description = "Using Azure storage blob container for pods(true)"
}

variable "initialModelName" {
  description = "Initial model folder name"
}

variable "domainName" {
  description = "Endpoint domain name"
}

#variable "TLScrtLocation" {
#  description = "TLS certificate .crt file path"
#}
#
#variable "TLSkeyLocation" {
#  description = "TLS certificate .key file path"
#}


