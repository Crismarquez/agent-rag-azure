
from azure.ai.projects import AIProjectClient
from azure.identity import ClientSecretCredential
from config.config import ENV_VARIABLES

from langchain_azure_ai.callbacks.tracers import AzureAIInferenceTracer

# Global cache variables
_tracer_instance = None


def get_tracer():
    """Get configured tracer - cached version """
    global _tracer_instance
    
    if _tracer_instance is not None:
        print("🔄 Using cached tracer")
        return _tracer_instance

    print("🔍 Creating tracer for the first time...")
    
    # Connect to Foundry Project
    project_connection_string = ENV_VARIABLES["AZURE_AI_FOUNDRY_PROJECT_CONNECTION_STRING"]
    project_client = AIProjectClient.from_connection_string(
        credential=ClientSecretCredential(
            client_id=ENV_VARIABLES["APP_REGISTRATION_CLIENT_ID"],
            client_secret=ENV_VARIABLES["APP_REGISTRATION_CLIENT_SECRET"],
            tenant_id=ENV_VARIABLES["APP_REGISTRATION_TENANT_ID"]
        ),
        conn_str=project_connection_string
    )
    
    application_insights_connection_string = project_client.telemetry.get_connection_string()
    
    _tracer_instance = AzureAIInferenceTracer(
        connection_string=application_insights_connection_string,
        enable_content_recording=True
    )

    print("✅ Tracer created and cached.")
    return _tracer_instance









