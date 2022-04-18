provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.resourceGrp}-${var.env}"
  location = var.location

  tags = {
    environment = var.env
  }
}


resource "azurerm_kubernetes_cluster" "cluster" {
  name                = var.aksClusterName
  location            = var.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = var.aksClusterName

  default_node_pool {
    name       = "default"
    vm_size    = var.clusterNodeSize
    min_count  = var.clusterMinNodeCount
    max_count  = var.clusterMaxNodeCount
    enable_auto_scaling = true

  }

  identity {
    type = "SystemAssigned"
  }
}

