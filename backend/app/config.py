import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class RuntimeConfig(BaseModel):
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"
    serper_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    sqlite_db_path: str = str(BASE_DIR / "data" / "agent_runs.db")
    faiss_index_dir: str = str(BASE_DIR / "data" / "faiss_index")

    @classmethod
    def from_env(cls, overrides: Optional[dict] = None) -> "RuntimeConfig":
        overrides = overrides or {}
        return cls(
            groq_api_key=overrides.get("groq_api_key") or os.getenv("GROQ_API_KEY") or None,
            groq_model=overrides.get("groq_model") or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
            serper_api_key=overrides.get("serper_api_key") or os.getenv("SERPER_API_KEY") or None,
            tavily_api_key=overrides.get("tavily_api_key") or os.getenv("TAVILY_API_KEY") or None,
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            sqlite_db_path=os.getenv("SQLITE_DB_PATH", str(BASE_DIR / "data" / "agent_runs.db")),
            faiss_index_dir=os.getenv("FAISS_INDEX_DIR", str(BASE_DIR / "data" / "faiss_index")),
        )
