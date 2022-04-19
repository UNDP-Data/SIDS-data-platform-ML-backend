import copy
import json

import yaml


def get_inputs(text: str, expect: [str] = None, default=None):
    ret = None
    while ret is None or ret == "":
        if expect is None:
            ret = input(text + " : ")
        else:
            while ret not in expect:
                ret = input(text + " " + str(expect) + " : ")

        if default is not None and ret is None or ret == "":
            ret = default
            break

    if ret is None:
        return default
    return ret


name = get_inputs("Please enter service name. (supported only [a-z, A-Z, -])")
model_folder = get_inputs("Model folder name?")
isSharedVolume = get_inputs("Do you need Azure Storage shared volume?", ["y", "n"])
if isSharedVolume == "y":
    share_vol_type = get_inputs("Type of Azure Storage shared volume?", ["file", "blob"])
    share_volume_name = get_inputs("Share volume name (share name / blob container name)?")

req_memory = get_inputs("Requested memory: [default 256Mi]", None, "256Mi")
req_cpu = get_inputs("Requested cpu: [default 100m]", None, "100m")

limit_memory = get_inputs("Memory limit: [default 1024Mi]", None, "1024Mi")
limit_cpu = get_inputs("CPU limit: [default 200m]", None, "200m")

with open('./deployment/service_template.json') as json_file:
    service_template = json.load(json_file)

with open('./deployment/deployment_template.json') as json_file:
    deployment_template = json.load(json_file)

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
        {"name": "azure", "mountPath": "/mnt/azure"}]

    deployment_template["spec"]["template"]["spec"]["volumes"] = [
      {
        "name": "azure",
        "csi": {
            "driver": share_vol_type + ".csi.azure.com",
            "volumeAttributes": {
                "secretName": "azure-secret",
                "mountOptions": '"-o allow_other --file-cache-timeout-in-seconds=120"'
            }
        }
      }
    ]

    if share_vol_type == "blob":
        deployment_template["spec"]["template"]["spec"]["volumes"][0]["csi"]["volumeAttributes"]["containerName"] = share_volume_name
    else:
        deployment_template["spec"]["template"]["spec"]["volumes"][0]["csi"]["volumeAttributes"][
            "shareName"] = share_volume_name


with open("./deployment/manifests/k8_keda_main.yml", 'r') as stream:
    data_loaded = yaml.unsafe_load_all(stream)
    for s in data_loaded:
        if s is None:
            break
        if s["kind"] == "Service" and s["metadata"]["name"] == name:
            raise Exception("Service name already exist")

    with open("./deployment/autogen_manifests/" + name + ".yml", 'w') as stream:
        yaml.safe_dump_all([service_template, deployment_template], stream)
        print("service deployment yml generated: " + "autogen_manifests/" + name + ".yml")

print("Updating ingress file")
# Read YAML file
with open("./deployment/manifests/ingress.yml", 'r') as stream:
    data_loaded = yaml.unsafe_load_all(stream)
    for s in data_loaded:

        add = True
        for p in s["spec"]["rules"][0]["http"]["paths"]:
            if p["path"] == "/" + model_folder + "/(.*)":
                print("WARNING!: Ingress route entry already found in the ingress.yml, Does not update")
                add = False
        # Assuming only one ingress

        if add:
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
            with open("./deployment/manifests/ingress.yml", 'w') as stream:
                yaml.safe_dump_all([s], stream)
                print("Ingress file updated")
        break

print("Adding auto scaler file")
# Read YAML file

existing_autoscaler = []
with open("./deployment/manifests/autoscaler.yml", 'r') as stream:
    data_loaded = yaml.unsafe_load_all(stream)
    not_exist = True
    template = None
    for s in data_loaded:
        if s is None:
            break

        existing_autoscaler.append(s)
        template = copy.deepcopy(s)
        if s["metadata"]["name"] == "autoscaler-" + name:
            not_exist = False
            print("WARNING!: Ingress route entry already found in the ingress.yml, Does not update")

    if not_exist:
        template["metadata"]["name"] = "autoscaler-" + name
        template["spec"]["scaleTargetRef"]["name"] = name
        existing_autoscaler.append(template)
        with open("./deployment/manifests/autoscaler.yml", 'w') as stream:
            yaml.safe_dump_all(existing_autoscaler, stream)
            print("Ingress file updated")
