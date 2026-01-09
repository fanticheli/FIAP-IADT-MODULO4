"""
Analisador de atividades usando MoViNet (Mobile Video Networks).
Detecta 600 acoes humanas do dataset Kinetics-600.
"""
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pathlib

from src.analyzers.base_analyzer import BaseAnalyzer
from src.models.detection import ActivityDetection
from src.utils.logger import logger


def _load_kinetics_600_labels() -> list[str]:
    """Carrega as 600 classes do Kinetics-600."""
    try:
        labels_path = tf.keras.utils.get_file(
            fname='kinetics_600_labels.txt',
            origin='https://raw.githubusercontent.com/tensorflow/models/f8af2291cced43fc9f1d9b41ddbf772ae7b0d7d2/official/projects/movinet/files/kinetics_600_labels.txt'
        )
        labels_path = pathlib.Path(labels_path)
        lines = labels_path.read_text().splitlines()
        return [line.strip() for line in lines]
    except Exception as e:
        logger.warning(f"Erro ao carregar labels: {e}")
        return []


# Mapeamento das principais classes do Kinetics-400 para portugues
# (as mais comuns/relevantes)
ACTIVITY_LABELS_PT = {
    "dancing": "Dancando",
    "dancing ballet": "Dancando ballet",
    "dancing gangnam style": "Dancando gangnam style",
    "breakdancing": "Breakdancing",
    "salsa dancing": "Dancando salsa",
    "robot dancing": "Dancando robo",
    "belly dancing": "Danca do ventre",
    "zumba": "Zumba",
    "applauding": "Aplaudindo",
    "clapping": "Batendo palmas",
    "waving hand": "Acenando",
    "hugging": "Abracando",
    "kissing": "Beijando",
    "shaking hands": "Apertando maos",
    "laughing": "Rindo",
    "crying": "Chorando",
    "yawning": "Bocejando",
    "sneezing": "Espirrando",
    "sticking tongue out": "Mostrando a lingua",
    "headbanging": "Headbanging",
    "singing": "Cantando",
    "beatboxing": "Beatbox",
    "whistling": "Assobiando",
    "talking": "Falando",
    "reading book": "Lendo livro",
    "reading newspaper": "Lendo jornal",
    "writing": "Escrevendo",
    "drawing": "Desenhando",
    "texting": "Digitando",
    "using computer": "Usando computador",
    "playing video games": "Jogando videogame",
    "using remote controller (not gaming)": "Usando controle remoto",
    "playing guitar": "Tocando violao",
    "playing piano": "Tocando piano",
    "playing drums": "Tocando bateria",
    "playing violin": "Tocando violino",
    "playing flute": "Tocando flauta",
    "playing accordion": "Tocando acordeao",
    "drumming fingers": "Tamborilar dedos",
    "cooking": "Cozinhando",
    "cooking egg": "Fritando ovo",
    "cooking chicken": "Cozinhando frango",
    "making a sandwich": "Fazendo sanduiche",
    "making pizza": "Fazendo pizza",
    "making sushi": "Fazendo sushi",
    "making tea": "Fazendo cha",
    "making a cake": "Fazendo bolo",
    "baking cookies": "Assando biscoitos",
    "eating": "Comendo",
    "eating burger": "Comendo hamburguer",
    "eating pizza": "Comendo pizza",
    "eating ice cream": "Comendo sorvete",
    "drinking": "Bebendo",
    "drinking beer": "Bebendo cerveja",
    "drinking shots": "Bebendo shots",
    "tasting food": "Provando comida",
    "running": "Correndo",
    "jogging": "Corrida leve",
    "walking": "Andando",
    "walking the dog": "Passeando com cachorro",
    "jumping": "Pulando",
    "jumping into pool": "Pulando na piscina",
    "swimming": "Nadando",
    "diving": "Mergulhando",
    "surfing water": "Surfando",
    "skiing": "Esquiando",
    "snowboarding": "Snowboard",
    "skateboarding": "Andando de skate",
    "riding a bike": "Andando de bicicleta",
    "motorcycling": "Andando de moto",
    "driving car": "Dirigindo",
    "exercising": "Exercitando",
    "doing aerobics": "Fazendo aerobica",
    "yoga": "Yoga",
    "stretching": "Alongando",
    "push up": "Flexao",
    "pull ups": "Barra",
    "situp": "Abdominal",
    "squat": "Agachamento",
    "deadlifting": "Levantamento terra",
    "bench pressing": "Supino",
    "punching bag": "Socando saco",
    "punching person (boxing)": "Boxeando",
    "wrestling": "Lutando",
    "playing basketball": "Jogando basquete",
    "playing soccer": "Jogando futebol",
    "playing tennis": "Jogando tenis",
    "playing volleyball": "Jogando volei",
    "golf": "Golfe",
    "bowling": "Boliche",
    "archery": "Arco e flecha",
    "cleaning": "Limpando",
    "cleaning floor": "Limpando chao",
    "cleaning windows": "Limpando janelas",
    "washing dishes": "Lavando louca",
    "washing hands": "Lavando maos",
    "ironing": "Passando roupa",
    "folding clothes": "Dobrando roupas",
    "making bed": "Arrumando cama",
    "brushing teeth": "Escovando dentes",
    "brushing hair": "Escovando cabelo",
    "cutting hair": "Cortando cabelo",
    "shaving": "Fazendo barba",
    "applying cream": "Passando creme",
    "doing nails": "Fazendo unhas",
    "taking a shower": "Tomando banho",
    "sleeping": "Dormindo",
    "sitting": "Sentado",
    "standing": "Em pe",
    "waiting in line": "Esperando na fila",
    "presenting": "Apresentando",
    "news anchoring": "Apresentando jornal",
    "testifying": "Testemunhando",
    "sign language interpreting": "Interpretando libras",
    "celebrating": "Comemorando",
    "opening present": "Abrindo presente",
    "blowing out candles": "Apagando velas",
    "throwing confetti": "Jogando confete",
}

# Classes do Kinetics-600 serao carregadas dinamicamente
KINETICS_600_LABELS = None


class ActivityAnalyzer(BaseAnalyzer):
    """
    Analisador de atividades usando MoViNet (Mobile Video Networks).

    Detecta 600 acoes humanas do dataset Kinetics-600 usando um modelo
    de video pre-treinado que considera contexto temporal.
    """

    # URL do modelo MoViNet-A2 (equilibrio entre precisao e velocidade)
    MODEL_URL = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"

    # Numero de frames para acumular antes de fazer predicao
    FRAMES_PER_PREDICTION = 8

    # Tamanho do frame para o modelo
    INPUT_SIZE = (224, 224)

    # Confianca minima para considerar uma acao detectada
    MIN_CONFIDENCE = 0.15

    def __init__(self, min_detection_confidence: float = 0.15):
        self._min_confidence = min_detection_confidence
        self._model = None
        self._model_signature = None
        self._labels = []
        self._activity_counter = 0
        self._frame_buffer = []
        self._last_prediction = None
        self._last_confidence = 0.0

    @property
    def name(self) -> str:
        return "activity_detection"

    def setup(self) -> None:
        """Inicializa o modelo MoViNet."""
        global KINETICS_600_LABELS

        self._activity_counter = 0
        self._frame_buffer = []
        self._last_prediction = None
        self._last_confidence = 0.0

        logger.info("Carregando modelo MoViNet (pode demorar na primeira vez)...")
        try:
            # Carrega o modelo
            self._model = hub.load(self.MODEL_URL)
            # Usa a signature para inferencia
            self._model_signature = self._model.signatures['serving_default']

            # Carrega as labels do Kinetics-600
            if KINETICS_600_LABELS is None:
                KINETICS_600_LABELS = _load_kinetics_600_labels()
            self._labels = KINETICS_600_LABELS

            logger.info(f"ActivityAnalyzer inicializado (MoViNet-A2, {len(self._labels)} classes)")
        except Exception as e:
            logger.error(f"Erro ao carregar MoViNet: {e}")
            logger.info("Usando fallback sem modelo de atividades")
            self._model = None
            self._model_signature = None

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocessa um frame para o modelo."""
        import cv2

        # Redimensiona para o tamanho esperado
        resized = cv2.resize(frame, self.INPUT_SIZE)

        # Converte BGR para RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normaliza para [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        return normalized

    def _predict_activity(self) -> tuple[str, float]:
        """
        Faz predicao de atividade usando os frames acumulados.

        Returns:
            Tupla (activity_name, confidence)
        """
        if self._model_signature is None or len(self._frame_buffer) < 2:
            return "unknown", 0.0

        try:
            # Empilha frames em um tensor [1, T, H, W, C]
            frames = np.stack(self._frame_buffer, axis=0)
            frames = np.expand_dims(frames, axis=0)  # Adiciona dimensao do batch

            # Converte para tensor TensorFlow
            input_tensor = tf.constant(frames, dtype=tf.float32)

            # Faz predicao usando a signature
            output = self._model_signature(image=input_tensor)
            logits = output['classifier_head'][0]

            # Aplica softmax para obter probabilidades
            probabilities = tf.nn.softmax(logits, axis=-1)

            # Pega a classe com maior probabilidade
            top_class = int(tf.argmax(probabilities, axis=-1).numpy())
            confidence = float(probabilities[top_class].numpy())

            # Mapeia para o nome da classe
            if self._labels and top_class < len(self._labels):
                activity = self._labels[top_class]
            else:
                activity = f"action_{top_class}"

            return activity, confidence

        except Exception as e:
            logger.debug(f"Erro na predicao: {e}")
            return "unknown", 0.0

    def _get_portuguese_label(self, activity: str) -> str:
        """Retorna o label em portugues para a atividade."""
        # Tenta encontrar traducao exata
        if activity in ACTIVITY_LABELS_PT:
            return ACTIVITY_LABELS_PT[activity]

        # Tenta encontrar traducao parcial
        for key, value in ACTIVITY_LABELS_PT.items():
            if key in activity or activity in key:
                return value

        # Formata o nome em ingles de forma legivel
        return activity.replace("_", " ").title()

    def analyze(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> list[ActivityDetection]:
        """
        Analisa atividades no frame.

        Args:
            frame: Frame em formato BGR (OpenCV).
            frame_number: Numero do frame.

        Returns:
            Lista de ActivityDetection encontradas.
        """
        detections = []

        if self._model_signature is None:
            self.setup()
            if self._model_signature is None:
                return detections

        # Preprocessa e adiciona ao buffer
        processed_frame = self._preprocess_frame(frame)
        self._frame_buffer.append(processed_frame)

        # Mantem apenas os ultimos N frames
        if len(self._frame_buffer) > self.FRAMES_PER_PREDICTION:
            self._frame_buffer.pop(0)

        # Faz predicao quando tiver frames suficientes
        if len(self._frame_buffer) >= self.FRAMES_PER_PREDICTION // 2:
            activity, confidence = self._predict_activity()

            # Atualiza ultima predicao se confianca for maior
            if confidence >= self._min_confidence:
                self._last_prediction = activity
                self._last_confidence = confidence

        # Retorna deteccao se tiver predicao valida
        if self._last_prediction and self._last_confidence >= self._min_confidence:
            self._activity_counter += 1

            detection = ActivityDetection(
                activity_id=self._activity_counter,
                activity=self._last_prediction,
                activity_label=self._get_portuguese_label(self._last_prediction),
                confidence=self._last_confidence,
                frame_number=frame_number,
                landmarks_velocity=0.0,
                pose_angles={},
                is_anomaly=False,
                anomaly_type=""
            )
            detections.append(detection)

        return detections

    def teardown(self) -> None:
        """Libera recursos."""
        self._model = None
        self._frame_buffer = []
        logger.info(f"ActivityAnalyzer finalizado. Total: {self._activity_counter} deteccoes")
