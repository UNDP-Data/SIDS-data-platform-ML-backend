resource "local_file" "kubeconfig_nginx" {
  count        = var.createAzureAppGW ? 0 : 1
  depends_on   = [azurerm_kubernetes_cluster.clusterNGINX[0]]
  filename     = "kubeconfig"
  content      = azurerm_kubernetes_cluster.clusterNGINX[0].kube_config_raw
}

resource "local_file" "kubeconfig_appgw" {
  count        = var.createAzureAppGW ? 1 : 0
  depends_on   = [azurerm_kubernetes_cluster.clusterAppGW[0]]
  filename     = "kubeconfig"
  content      = azurerm_kubernetes_cluster.clusterAppGW[0].kube_config_raw
}
