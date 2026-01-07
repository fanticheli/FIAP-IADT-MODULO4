"""
Sistema de logging configurado para a aplicação.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "video_analysis",
    level: int = logging.INFO,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configura e retorna um logger.

    Args:
        name: Nome do logger.
        level: Nível de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Caminho opcional para arquivo de log.

    Returns:
        Logger configurado.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Evitar duplicação de handlers
    if logger.handlers:
        return logger

    # Formato do log
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Handler para arquivo (opcional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Logger padrão da aplicação
logger = setup_logger()
