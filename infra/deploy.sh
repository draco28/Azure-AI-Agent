#!/bin/bash

set -e

# ─── Variables ───────────────────────────────────────────────
RESOURCE_GROUP="ai-agent-rg"
LOCATION="eastus"
ACR_NAME="aiagentacr"
CONTAINER_APP_ENV="ai-agent-env"
LOG_ANALYTICS_WORKSPACE="ai-agent-logs"
KEY_VAULT_NAME="ai-agent-kv"
POSTGRES_SERVER="ai-agent-postgres"
POSTGRES_DB="azure_ai_agent"
POSTGRES_ADMIN="dbadmin"
IMAGE_NAME="azure-ai-agent"
IMAGE_TAG="${1:-latest}"  # pass tag as first argument, default to 'latest'

# ─── 1. Resource Group ──────────────────────────────────────
echo "Creating resource group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# ─── 2. Azure Container Registry ────────────────────────────
echo "Creating container registry..."
az acr create --name $ACR_NAME --resource-group $RESOURCE_GROUP --sku Basic

# Build and push Docker image to ACR
az acr login --name $ACR_NAME
docker build -t $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG .
docker push $ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG

# ─── 3. Log Analytics Workspace ─────────────────────────────
echo "Creating Log Analytics workspace..."
az monitor log-analytics workspace create \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $LOG_ANALYTICS_WORKSPACE

# Fetch workspace ID and key for Container Apps Environment
LOG_ANALYTICS_ID=$(az monitor log-analytics workspace show \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $LOG_ANALYTICS_WORKSPACE \
  --query customerId -o tsv)

LOG_ANALYTICS_KEY=$(az monitor log-analytics workspace get-shared-keys \
  --resource-group $RESOURCE_GROUP \
  --workspace-name $LOG_ANALYTICS_WORKSPACE \
  --query primarySharedKey -o tsv)

# ─── 4. Key Vault ───────────────────────────────────────────
echo "Creating Key Vault..."
az keyvault create \
  --name $KEY_VAULT_NAME \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION

# Store external API keys (you'll set actual values before running)
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "glm-api-key" --value "${GLM_API_KEY:?Set GLM_API_KEY env var}"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "glm-api-base" --value "${GLM_API_BASE:?Set GLM_API_BASE env var}"
az keyvault secret set --vault-name $KEY_VAULT_NAME --name "langsmith-api-key" --value "${LANGSMITH_API_KEY:?Set LANGSMITH_API_KEY env var}"

# ─── 5. Azure Database for PostgreSQL ───────────────────────
echo "Creating PostgreSQL flexible server..."
az postgres flexible-server create \
  --name $POSTGRES_SERVER \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --admin-user $POSTGRES_ADMIN \
  --admin-password "${POSTGRES_PASSWORD:?Set POSTGRES_PASSWORD env var}" \
  --sku-name Standard_B1ms \
  --storage-size 32 \
  --version 16

# Create the application database
az postgres flexible-server db create \
  --resource-group $RESOURCE_GROUP \
  --server-name $POSTGRES_SERVER \
  --database-name $POSTGRES_DB

# ─── 6. Container Apps Environment ──────────────────────────
echo "Creating Container Apps environment..."
az containerapp env create \
  --name $CONTAINER_APP_ENV \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --logs-workspace-id $LOG_ANALYTICS_ID \
  --logs-workspace-key $LOG_ANALYTICS_KEY

# ─── 7. Container Apps ──────────────────────────────────────
FULL_IMAGE="$ACR_NAME.azurecr.io/$IMAGE_NAME:$IMAGE_TAG"

echo "Deploying API container..."
az containerapp create \
  --name ai-agent-api \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $FULL_IMAGE \
  --target-port 8000 \
  --ingress external \
  --registry-server $ACR_NAME.azurecr.io \
  --system-assigned \
  --min-replicas 1 \
  --max-replicas 3 \
  --command "/app/.venv/bin/uvicorn" "src.api.app:app" "--host" "0.0.0.0" "--port" "8000"

echo "Deploying UI container..."
az containerapp create \
  --name ai-agent-ui \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $FULL_IMAGE \
  --target-port 8080 \
  --ingress external \
  --registry-server $ACR_NAME.azurecr.io \
  --system-assigned \
  --min-replicas 1 \
  --max-replicas 3 \
  --command "/app/.venv/bin/chainlit" "run" "src/ui/app.py" "--host" "0.0.0.0" "--port" "8080"

echo "Deploying MCP container..."
az containerapp create \
  --name ai-agent-mcp \
  --resource-group $RESOURCE_GROUP \
  --environment $CONTAINER_APP_ENV \
  --image $FULL_IMAGE \
  --target-port 8001 \
  --ingress internal \
  --registry-server $ACR_NAME.azurecr.io \
  --system-assigned \
  --min-replicas 1 \
  --max-replicas 3 \
  --command "/app/.venv/bin/python" "-m" "src.mcp.server"

# ─── 8. Managed Identity — Role Assignments ─────────────────
echo "Configuring managed identity access..."

# Get identity IDs for each container app
API_IDENTITY=$(az containerapp show --name ai-agent-api --resource-group $RESOURCE_GROUP --query identity.principalId -o tsv)
UI_IDENTITY=$(az containerapp show --name ai-agent-ui --resource-group $RESOURCE_GROUP --query identity.principalId -o tsv)
MCP_IDENTITY=$(az containerapp show --name ai-agent-mcp --resource-group $RESOURCE_GROUP --query identity.principalId -o tsv)

# Grant Key Vault access to all containers
for IDENTITY in $API_IDENTITY $UI_IDENTITY $MCP_IDENTITY; do
  az role assignment create \
    --assignee $IDENTITY \
    --role "Key Vault Secrets User" \
    --scope $(az keyvault show --name $KEY_VAULT_NAME --query id -o tsv)
done

# Grant ACR pull access
for IDENTITY in $API_IDENTITY $UI_IDENTITY $MCP_IDENTITY; do
  az role assignment create \
    --assignee $IDENTITY \
    --role "AcrPull" \
    --scope $(az acr show --name $ACR_NAME --query id -o tsv)
done

echo ""
echo "Deployment complete!"
echo "API URL: $(az containerapp show --name ai-agent-api --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)"
echo "UI URL:  $(az containerapp show --name ai-agent-ui --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn -o tsv)"
