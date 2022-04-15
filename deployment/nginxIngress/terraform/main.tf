provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = var.resource_grp
  location = "eastus"

  tags = {
    environment = "staging"
  }
}

resource "azurerm_storage_account" "sa" {
  name                     = "mlbackendstorage2"
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Premium"
  account_replication_type = "LRS"

  tags = {
    environment = "staging"
  }

}

resource "azurerm_storage_container" "dataset" {
  name                  = "datasets"
  storage_account_name  = azurerm_storage_account.sa.name
  container_access_type = "blob"
}

resource "azurerm_container_registry" "acr" {
  name                = "acrmlbackend2"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
}

resource "azurerm_kubernetes_cluster" "cluster" {
  name                = "sidMLBackendCluster2"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "sidMLBackendCluster"

  default_node_pool {
    name       = "default"
    node_count = "1"
    vm_size    = "standard_d2_v2"
    min_count  = "1"
    max_count  = "3"
    enable_auto_scaling = true

  }

  identity {
    type = "SystemAssigned"
  }
}


#provider "kubernetes" {
#  config_path    = "./kubeconfig"
#}
#
#resource "kubernetes_namespace" "ns" {
#  metadata {
#    annotations = {
#      name = "ml-app"
#    }
#    name = "ml-app"
#  }
#}
