provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = var.resourceGrp
  location = var.location

  tags = {
    environment = var.env
  }
}

resource "azurerm_log_analytics_workspace" "logw" {
    # The WorkSpace name has to be unique across the whole of azure, not just the current subscription/tenant.
    name                = "${var.appName}-log-workspace"
    location            = var.location
    resource_group_name = azurerm_resource_group.rg.name
}


resource "azurerm_kubernetes_cluster" "clusterNGINX" {
  count               = var.createAzureAppGW ? 0: 1
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

   oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.logw.id
   }

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_kubernetes_cluster" "clusterAppGW" {
  count               = var.createAzureAppGW ? 1: 0
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

    ingress_application_gateway  {
        subnet_cidr                = "10.2.0.0/16"
        gateway_name               = "${var.aksClusterName}-AGIC"
    }

     oms_agent {
      log_analytics_workspace_id = azurerm_log_analytics_workspace.logw.id
     }

  identity {
    type = "SystemAssigned"
  }
}

#resource "azurerm_kubernetes_cluster_node_pool" "survey" {
#  name                  = "internal"
#  kubernetes_cluster_id = azurerm_kubernetes_cluster.clusterNGINX[0].id
#  vm_size               = "Standard_DS2_v2"
#  node_count            = 1
#
#  tags = {
#    Environment = "Testing"
#  }
#}