resource "azurerm_storage_account" "sa" {
  depends_on               = [azurerm_resource_group.rg]
  name                     = var.storageAccountName
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = var.location
  account_tier             = "Standard"
  account_replication_type = "LRS"

  tags = {
    environment = var.env
  }

}

resource "azurerm_storage_container" "container" {
  depends_on            = [azurerm_storage_account.sa]
  name                  = var.sharedDataContainerName
  storage_account_name  = azurerm_storage_account.sa.name
  container_access_type = "container"
}
