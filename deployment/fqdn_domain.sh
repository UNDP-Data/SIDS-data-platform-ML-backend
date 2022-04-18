# Public IP address of your ingress controller
IP=52.226.231.219

# Name to associate with public IP address
DNSNAME="ml-aks-ingress"

# Get the resource-id of the public ip
PUBLICIPID=$(az network public-ip list --query "[?ipAddress!=null]|[?contains(ipAddress, '$IP')].[id]" --output tsv)

# Update public ip address with DNS name
az network public-ip update --ids $PUBLICIPID --dns-name $DNSNAME

# Display the FQDN
az network public-ip show --ids $PUBLICIPID --query "[dnsSettings.fqdn]" --output tsv

#openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
#    -out aks-ingress-tls.crt \
#    -keyout aks-ingress-tls.key \
#    -subj "/CN=ml-aks-ingress.eastus.cloudapp.azure.com/O=aks-ingress-tls"