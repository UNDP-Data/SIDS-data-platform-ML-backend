import json
import sys
import yaml

i = 1
name = sys.argv[i]
i = i+1
model_folder = sys.argv[i]
i = i+1
namespace = sys.argv[i]
i = i+1
ingress_tls_secret = sys.argv[i]
i = i+1
isSharedVolume = sys.argv[i]
i = i+1
if isSharedVolume == "y":
    share_vol_type = sys.argv[i]
    i = i + 1
    share_volume_name = sys.argv[i]
    i = i + 1

req_memory = sys.argv[i]
i = i+1
req_cpu = sys.argv[i]
i = i+1

limit_memory = sys.argv[i]
i = i+1
limit_cpu = sys.argv[i]
i = i+1
isAppGW = sys.argv[i]

service_template = None
deployment_template = None
remaining_templates = []

with open("./deployment/manifests/k8_keda_main.yml", 'r') as stream:
    data_loaded = yaml.unsafe_load_all(stream)
    for s in data_loaded:
        if s is None:
            break
        if s["kind"] == "Service":
            service_template = s
            with open('./deployment/service_template.json', 'w') as outfile:
                json.dump(s, outfile)
        elif s["kind"] == "Deployment":
            deployment_template = s
            with open('./deployment/deployment_template.json', 'w') as outfile:
                json.dump(s, outfile)
        else:
            remaining_templates.append(s)


print("Updating service and deployment file")

service_template["metadata"]["name"] = name
service_template["spec"]["selector"]["app"] = name
service_template["spec"]["type"] = "ClusterIP"

deployment_template["metadata"]["name"] = name
deployment_template["metadata"]["labels"]["app"] = name
deployment_template["spec"]["selector"]["matchLabels"]["app"] = name
deployment_template["spec"]["template"]["metadata"]["labels"]["app"] = name
deployment_template["spec"]["template"]["spec"]["containers"][0]["name"] = name
deployment_template["spec"]["template"]["spec"]["containers"][0]["resources"] = {
    "limits": {
        "cpu": "",
        "memory": ""
    },
    "requests": {
        "cpu": "",
        "memory": ""
    }
}
deployment_template["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["cpu"] = limit_cpu
deployment_template["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["memory"] = limit_memory

deployment_template["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]["cpu"] = req_cpu
deployment_template["spec"]["template"]["spec"]["containers"][0]["resources"]["requests"]["memory"] = req_memory
deployment_template["spec"]["template"]["spec"]["containers"][0]["env"].append({
    "name": "MODEL_SERVICE",
    "value": model_folder
})

if isSharedVolume == "y":
    deployment_template["spec"]["template"]["spec"]["containers"][0]["volumeMounts"] = [
        {"name": "azure", "mountPath": "/mnt/azure/" + share_volume_name}]

    deployment_template["spec"]["template"]["spec"]["volumes"] = [
      {
        "name": "azure",
        "csi": {
            "driver": share_vol_type + ".csi.azure.com",
            "volumeAttributes": {
                "secretName": "azure-secret",
                "mountOptions": "-o allow_other --file-cache-timeout-in-seconds=120"
            }
        }
      }
    ]

    if share_vol_type == "blob":
        deployment_template["spec"]["template"]["spec"]["volumes"][0]["csi"]["volumeAttributes"]["containerName"] = share_volume_name
    else:
        deployment_template["spec"]["template"]["spec"]["volumes"][0]["csi"]["volumeAttributes"][
            "shareName"] = share_volume_name

remaining_templates.append(service_template)
remaining_templates.append(deployment_template)

with open("./deployment/manifests/k8_keda_main.yml", 'w') as stream:
    yaml.safe_dump_all(remaining_templates, stream)
    print("service deployment yml generated: " + "./deployment/manifests/k8_keda_main.yml")

print("Updating ingress file 1")
# Read YAML file

print("Application gateway created: " + isAppGW)
if isAppGW == "true":
    s = {
      "apiVersion": "networking.k8s.io/v1",
      "kind": "Ingress",
      "metadata": {
        "name": "sidmlbackend-http-ingress",
        "namespace": namespace,
        "annotations": {
          "kubernetes.io/ingress.class": "azure/application-gateway",
          # "appgw.ingress.kubernetes.io/appgw-ssl-certificate": "httpCert",
          "appgw.ingress.kubernetes.io/ssl-redirect": "true"
        }
      },
      "spec": {
        "rules": [
          {
            "http": {
              "paths": [
              ]
            }
          }
        ]
      }
    }
else:
    s = {
      "apiVersion": "networking.k8s.io/v1",
      "kind": "Ingress",
      "metadata": {
        "annotations": {
          "nginx.ingress.kubernetes.io/proxy-body-size": "0",
          "nginx.ingress.kubernetes.io/proxy-read-timeout": "600",
          "nginx.ingress.kubernetes.io/proxy-send-timeout": "600",
          "nginx.ingress.kubernetes.io/use-regex": "true"
        },
        "name": "sidmlbackend-ingress",
        "namespace": namespace
      },
      "spec": {
        "ingressClassName": "nginx",
        "rules": [
          {
            # "host": "",
            "http": {
                "paths": []
            }
          }
        ]
        #   ,
        # "tls": [
        #   {
        #     "secretName": "aks-ingress-tls",
        #     "hosts": []
        #   }
        # ]
      }
    }
s["metadata"]["name"] = "ingress-" + name
# s["spec"]["rules"][0]["host"] = domain

s["spec"]["rules"][0]["http"]["paths"].append({
      "path": "/" + model_folder + "/(.*)",
      "pathType": "Prefix",
      "backend": {
        "service": {
          "name": name,
          "port": {
            "number": 80
          }
        }
      }
    })
s["spec"]["rules"][0]["http"]["paths"].append({
      "path": "/(.*)",
      "pathType": "Prefix",
      "backend": {
        "service": {
          "name": name,
          "port": {
            "number": 80
          }
        }
      }
    })
# s["spec"]["tls"][0]["hosts"].append(domain)
# s["spec"]["tls"][0]["secretName"] = ingress_tls_secret

with open("./deployment/manifests/ingress.yml", 'w') as stream:
    yaml.safe_dump_all([s], stream)
    print("Ingress file updated")


with open("./deployment/manifests/autoscaler.yml", 'r') as stream:
    data_loaded = yaml.unsafe_load_all(stream)
    not_exist = True
    template = None
    for s in data_loaded:
        if s is None:
            break

        s["metadata"]["name"] = "autoscaler-" + name
        s["spec"]["scaleTargetRef"]["name"] = name
        with open("./deployment/manifests/autoscaler.yml", 'w') as stream:
            yaml.safe_dump_all([s], stream)
            print("Auto scale file updated")
            break
