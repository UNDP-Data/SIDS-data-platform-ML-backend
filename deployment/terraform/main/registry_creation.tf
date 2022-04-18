resource "azurerm_container_registry" "acr" {
  depends_on          = [azurerm_resource_group.rg]
  name                = var.acrName
  resource_group_name = azurerm_resource_group.rg.name
  location            = var.location
  sku                 = "Basic"
  admin_enabled       = true
}