"""配置管理"""
import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    """HelloAgents配置类 - 默认DeepSeek，支持外部配置"""

    # ---------- LLM 通用配置（默认DeepSeek）----------
    default_model: str = "deepseek-chat"
    default_provider: str = "deepseek"
    temperature: float = 0.7
    max_tokens: Optional[int] = None

    # ---------- DeepSeek 专属配置 ----------
    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"

    # ---------- OpenAI 配置（兼容，便于切换）----------
    openai_api_key: Optional[str] = None
    openai_base_url: Optional[str] = None

    # ---------- 系统配置 ----------
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100

    @classmethod
    def from_env(cls) -> "Config":
        """从环境变量创建配置，未设置则使用默认值（DeepSeek）"""
        return cls(
            # 通用LLM配置（可通过环境变量覆盖）
            default_model=os.getenv("DEFAULT_MODEL", "deepseek-chat"),
            default_provider=os.getenv("DEFAULT_PROVIDER", "deepseek"),
            temperature=float(os.getenv("TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("MAX_TOKENS")) if os.getenv("MAX_TOKENS") else None,

            # DeepSeek 配置
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),

            # OpenAI 配置（备选）
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),

            # 系统配置
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_history_length=int(os.getenv("MAX_HISTORY_LENGTH", "100")),
        )

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.dict()