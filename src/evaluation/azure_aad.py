import datetime
from azure.identity import DefaultAzureCredential
from azure.core.credentials import AccessToken

default_credential = DefaultAzureCredential()

def get_or_refresh_token(token=None) -> AccessToken.token:
    """Refresh AAD token"""
    # Check if Azure token is still valid
    if not token or datetime.datetime.fromtimestamp(token.expires_on) < datetime.datetime.now():
        token = default_credential.get_token("https://cognitiveservices.azure.com")
    return token.token