import streamlit as st
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_url: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow",
    )


def get_api_url() -> str:
    settings = Settings()

    try:
        secret_api_url = st.secrets.get("API_URL")
    except FileNotFoundError:
        secret_api_url = None

    api_url = secret_api_url or settings.api_url
    if not isinstance(api_url, str) or not api_url.strip():
        raise ValueError(
            "API_URL is not configured. Add API_URL to Streamlit Cloud "
            "Settings > Secrets, or set it in your local .env file."
        )

    return api_url.strip()
