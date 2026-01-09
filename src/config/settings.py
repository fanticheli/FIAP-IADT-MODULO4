"""
Configuracoes globais da aplicacao de analise de video.
"""
from pathlib import Path


class Settings:
    """Configuracoes centralizadas do projeto."""

    # Diretorios base
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    VIDEOS_DIR = BASE_DIR / "videos"
    OUTPUT_DIR = BASE_DIR / "resultado"

    # Configuracoes de processamento de video
    FRAME_SAMPLE_RATE = 1  # Processar 1 a cada N frames (1 = todos)

    # Configuracoes de visualizacao
    FACE_BOX_COLOR = (0, 255, 0)  # Verde (BGR)
    FACE_BOX_THICKNESS = 2
    FACE_LABEL_FONT_SCALE = 0.6
    FACE_LABEL_COLOR = (255, 255, 255)  # Branco (BGR)

    # Configuracoes de saida
    OUTPUT_VIDEO_CODEC = "mp4v"
    OUTPUT_VIDEO_FPS = None  # None = usar FPS original do video


# Instancia singleton para uso global
settings = Settings()
