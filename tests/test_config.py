from unittest.mock import patch
from config import Settings


def test_settings_defaults():
    """Verify default configuration settings load properly when no custom env is provided."""
    with patch.dict("os.environ", {}, clear=True):
        settings = Settings(_env_file=None)
        assert settings.APP_NAME == "CodeLearn AI"
        assert settings.APP_ENV == "development"
        assert settings.APP_DEBUG is True
        assert settings.PORT == 8000
        assert settings.HOST == "0.0.0.0"
        assert "postgresql://" in settings.DATABASE_URL
        assert settings.MAX_REPOSITORY_SIZE_MB == 50
        assert settings.GROQ_RPM_LIMIT == 30
        assert settings.GEMINI_RPM_LIMIT == 15


def test_settings_custom_env():
    """Verify environment variable overrides correctly populate Settings fields."""
    custom_env = {
        "APP_NAME": "Custom CodeLearn",
        "APP_ENV": "production",
        "APP_DEBUG": "false",
        "PORT": "9000",
        "MAX_REPOSITORY_SIZE_MB": "100",
        "GROQ_MODEL": "llama-3.3-70b-versatile",
    }
    with patch.dict("os.environ", custom_env, clear=True):
        settings = Settings(_env_file=None)
        assert settings.APP_NAME == "Custom CodeLearn"
        assert settings.APP_ENV == "production"
        assert settings.APP_DEBUG is False
        assert settings.PORT == 9000
        assert settings.MAX_REPOSITORY_SIZE_MB == 100
        assert settings.GROQ_MODEL == "llama-3.3-70b-versatile"


def test_database_url_assembly():
    """Verify database connection URL is properly assembled from decomposed POSTGRES_* variables."""
    custom_db_env = {
        "POSTGRES_USER": "test_user",
        "POSTGRES_PASSWORD": "test_password_123",
        "POSTGRES_HOST": "db.example.com",
        "POSTGRES_PORT": "5433",
        "POSTGRES_DB": "test_db",
    }
    with patch.dict("os.environ", custom_db_env, clear=True):
        settings = Settings(_env_file=None)
        assert "postgresql://test_user:test_password_123@db.example.com:5433/test_db" == settings.DATABASE_URL
