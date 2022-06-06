provider "kubernetes" {
  config_path    = "./kubeconfig"
}

resource "kubernetes_namespace" "ns_nginx" {
  count          = var.createAzureAppGW ? 0: 1
  depends_on     = [local_file.kubeconfig_nginx]
  metadata {
    annotations = {
      name = var.appNamespace
    }
    name = var.appNamespace
  }
}

resource "kubernetes_namespace" "ns_appgw" {
  count          = var.createAzureAppGW ? 1: 0
  depends_on     = [local_file.kubeconfig_appgw]
  metadata {
    annotations = {
      name = var.appNamespace
    }
    name = var.appNamespace
  }
}

resource "azurerm_role_assignment" "aksacrrole_nginx" {
  count                            = var.createAzureAppGW ? 0 : 1
  principal_id                     = azurerm_kubernetes_cluster.clusterNGINX[0].kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.acr.id
  skip_service_principal_aad_check = true
}

resource "azurerm_role_assignment" "aksacrrole_appgw" {
  count                            = var.createAzureAppGW ? 1 : 0
  principal_id                     = azurerm_kubernetes_cluster.clusterAppGW[0].kubelet_identity[0].object_id
  role_definition_name             = "AcrPull"
  scope                            = azurerm_container_registry.acr.id
  skip_service_principal_aad_check = true
}

resource "kubernetes_secret" "storage-secret" {
  metadata {
    name      = "azure-secret"
    namespace = var.appNamespace
  }

  data = {
    "azurestorageaccountname": var.storageAccountName
    "azurestorageaccountkey": azurerm_storage_account.sa.primary_access_key
  }
}

resource "kubernetes_secret" "storageaccountconnectionstring" {
  metadata {
    name      = "storageaccountconnectionstring"
    namespace = var.appNamespace
  }

  data = {
    "storageAccountConnectionString": azurerm_storage_account.sa.primary_connection_string
  }
}

resource "null_resource" "deploy_export" {
  triggers = {
    always_run = "${timestamp()}"
  }
  provisioner "local-exec" {
      command = "cd ../../ && func kubernetes deploy --name ${var.appName} --registry ${azurerm_container_registry.acr.name}.azurecr.io  --namespace ${var.appNamespace} --dry-run > ./deployment/manifests/k8_keda_main.yml"
  }
}

resource "null_resource" "update_k8_file" {
  triggers = {
    "after": null_resource.deploy_export.id
  }
  provisioner "local-exec" {
      command = "cd ../../ && python ./deployment/initial_service_update.py ${var.appName}-http ${var.initialModelName} ${var.appNamespace} aks-ingress-tls y blob ${var.sharedDataContainerName} 1024Mi 300m 2048Mi 800m ${var.createAzureAppGW}"
  }
}

provider "helm" {
  kubernetes {
    config_path = "./kubeconfig"
  }
}

resource "helm_release" "blobDriver" {
  count      = var.useBlobStorage ? 1 : 0
  name       = "blob-csi-driver"

  repository = "https://raw.githubusercontent.com/kubernetes-sigs/blob-csi-driver/master/charts"
  chart      = "blob-csi-driver"

  set {
    name  = "node.enableBlobfuseProxy"
    value = "true"
  }

  set {
    name  = "namespace"
    value = "kube-system"
  }

  set {
    name  = "version"
    value = "v1.10.0"
  }
}

resource "helm_release" "nginx_ingress" {
  name       = "nginx-ingress-controller"
  repository = "https://charts.bitnami.com/bitnami"
  chart      = "nginx-ingress-controller"
  namespace = var.appNamespace

}

resource "helm_release" "traefik_ingress" {
  name       = "traefik-ingress-controller"
  repository = "https://helm.traefik.io/traefik"
  chart      = "traefik"
  namespace = var.appNamespace
}
