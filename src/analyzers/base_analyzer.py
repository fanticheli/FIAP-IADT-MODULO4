"""
Classe base abstrata para analisadores de vídeo.
"""
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseAnalyzer(ABC):
    """
    Classe base para todos os analisadores.

    Implementa o padrão Strategy, permitindo que diferentes
    tipos de análise sejam aplicados de forma intercambiável.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome identificador do analisador."""
        pass

    @abstractmethod
    def analyze(self, frame: np.ndarray, frame_number: int) -> list[Any]:
        """
        Analisa um frame e retorna as detecções encontradas.

        Args:
            frame: Frame do vídeo em formato numpy array (BGR).
            frame_number: Número sequencial do frame.

        Returns:
            Lista de detecções encontradas no frame.
        """
        pass

    def setup(self) -> None:
        """
        Método opcional para inicialização do analisador.
        Chamado uma vez antes do processamento iniciar.
        """
        pass

    def teardown(self) -> None:
        """
        Método opcional para limpeza do analisador.
        Chamado uma vez após o processamento terminar.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
