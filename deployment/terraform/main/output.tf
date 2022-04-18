resource "local_file" "kubeconfig" {
  depends_on   = [azurerm_kubernetes_cluster.cluster]
  filename     = "kubeconfig"
  content      = azurerm_kubernetes_cluster.cluster.kube_config_raw
}

resource "local_file" "ba" {
  depends_on = [azurerm_storage_container.container]
  filename = "connectionstring"
  content = azurerm_storage_account.sa.primary_access_key
}

output "connetion_string" {
  value = azurerm_storage_account.sa.primary_connection_string
  description = "Create storage connection string"
  sensitive = true
}