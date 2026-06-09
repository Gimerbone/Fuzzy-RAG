import os


class Config:
    POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", 5432))
    POSTGRES_USER = os.environ.get("POSTGRES_USER", "fuzzyrag")
    POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "fuzzyrag")
    POSTGRES_DB = os.environ.get("POSTGRES_DB", "fuzzyrag")

    LLM_MODEL = os.environ.get("LLM_MODEL", "StanfordAIMI/CheXagent-8b")
    LOAD_BITS = int(os.environ.get("LOAD_BITS", 4))  # 4 or 8 for bitsandbytes

    EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL",
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    )
    VECTOR_DIM = int(os.environ.get("VECTOR_DIM", 512))

    RRF_K = int(os.environ.get("RRF_K", 60))
    TOP_K = int(os.environ.get("TOP_K", 5))

    RADQA_DATA_PATH = os.environ.get("RADQA_DATA_PATH", "./data/radqa")

    @property
    def dsn(self) -> str:
        return (
            f"host={self.POSTGRES_HOST} port={self.POSTGRES_PORT} "
            f"dbname={self.POSTGRES_DB} user={self.POSTGRES_USER} "
            f"password={self.POSTGRES_PASSWORD}"
        )


config = Config()
