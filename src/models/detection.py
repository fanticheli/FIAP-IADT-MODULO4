"""
Modelos de dados para detecções e análises.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BoundingBox:
    """Representa uma caixa delimitadora."""
    top: int
    right: int
    bottom: int
    left: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)

    def to_tuple(self) -> tuple[int, int, int, int]:
        """Retorna (top, right, bottom, left)."""
        return (self.top, self.right, self.bottom, self.left)

    def to_opencv(self) -> tuple[int, int, int, int]:
        """Retorna (x, y, width, height) para OpenCV."""
        return (self.left, self.top, self.width, self.height)


@dataclass(frozen=True)
class FaceDetection:
    """Representa uma detecção de rosto em um frame."""
    face_id: int
    bounding_box: BoundingBox
    frame_number: int
    confidence: Optional[float] = None

    def __repr__(self) -> str:
        return f"Face(id={self.face_id}, frame={self.frame_number})"


@dataclass(frozen=True)
class EmotionDetection:
    """Representa uma detecção de emoção em um frame."""
    emotion_id: int
    emotion: str  # angry, happy, sad, etc
    emotion_label: str  # Raiva, Feliz, Triste, etc
    confidence: float
    frame_number: int
    box: tuple[int, int, int, int]  # x, y, w, h
    all_emotions: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Emotion({self.emotion_label}, conf={self.confidence:.2f})"


@dataclass(frozen=True)
class ActivityDetection:
    """Representa uma detecção de atividade em um frame."""
    activity_id: int
    activity: str  # standing, sitting, walking, etc
    activity_label: str  # Em pé, Sentado, Andando, etc
    confidence: float
    frame_number: int
    # Dados para detecção de anomalias
    landmarks_velocity: float = 0.0  # Velocidade media dos landmarks
    pose_angles: dict = field(default_factory=dict)  # Angulos corporais
    is_anomaly: bool = False  # Se este frame contem movimento anomalo
    anomaly_type: str = ""  # Tipo de anomalia detectada

    def __repr__(self) -> str:
        return f"Activity({self.activity_label}, conf={self.confidence:.2f})"


@dataclass
class FrameAnalysis:
    """Resultado da análise de um frame."""
    frame_number: int
    timestamp_ms: float
    faces: list[FaceDetection] = field(default_factory=list)
    emotions: list[EmotionDetection] = field(default_factory=list)
    activities: list[ActivityDetection] = field(default_factory=list)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    @property
    def emotion_count(self) -> int:
        return len(self.emotions)

    @property
    def activity_count(self) -> int:
        return len(self.activities)

    def add_face(self, face: FaceDetection) -> None:
        self.faces.append(face)

    def add_emotion(self, emotion: EmotionDetection) -> None:
        self.emotions.append(emotion)

    def add_activity(self, activity: ActivityDetection) -> None:
        self.activities.append(activity)


@dataclass
class VideoAnalysisResult:
    """Resultado completo da análise de um vídeo."""
    video_path: str
    total_frames: int
    processed_frames: int
    fps: float
    duration_seconds: float
    frame_analyses: list[FrameAnalysis] = field(default_factory=list)

    @property
    def total_faces_detected(self) -> int:
        """Total de detecções de rosto (pode contar o mesmo rosto várias vezes)."""
        return sum(fa.face_count for fa in self.frame_analyses)

    @property
    def frames_with_faces(self) -> int:
        """Número de frames que contêm pelo menos um rosto."""
        return sum(1 for fa in self.frame_analyses if fa.face_count > 0)

    @property
    def total_emotions_detected(self) -> int:
        """Total de detecções de emoção."""
        return sum(fa.emotion_count for fa in self.frame_analyses)

    @property
    def total_activities_detected(self) -> int:
        """Total de detecções de atividade."""
        return sum(fa.activity_count for fa in self.frame_analyses)

    def get_emotion_counts(self) -> dict:
        """Conta ocorrências de cada emoção."""
        counts = {}
        for fa in self.frame_analyses:
            for emotion in fa.emotions:
                label = emotion.emotion_label
                counts[label] = counts.get(label, 0) + 1
        return counts

    def get_activity_counts(self) -> dict:
        """Conta ocorrências de cada atividade."""
        counts = {}
        for fa in self.frame_analyses:
            for activity in fa.activities:
                label = activity.activity_label
                counts[label] = counts.get(label, 0) + 1
        return counts

    def get_face_anomalies(self) -> list[dict]:
        """Detecta anomalias de reconhecimento facial."""
        anomalies = []

        if len(self.frame_analyses) < 2:
            return anomalies

        for i in range(1, len(self.frame_analyses)):
            prev = self.frame_analyses[i - 1]
            curr = self.frame_analyses[i]
            diff = curr.face_count - prev.face_count

            # Mudança brusca: diferença >= 2 rostos
            if abs(diff) >= 2:
                anomalies.append({
                    "tipo": "Mudança brusca",
                    "frame": curr.frame_number,
                    "descricao": f"De {prev.face_count} para {curr.face_count} rostos"
                })

        return anomalies

    def get_emotion_anomalies(self) -> list[dict]:
        """Detecta anomalias de emoções."""
        anomalies = []

        if len(self.frame_analyses) < 2:
            return anomalies

        # Emoções contrastantes (mudança brusca)
        contrasting = {
            "happy": ["angry", "sad", "fear"],
            "angry": ["happy"],
            "sad": ["happy", "surprise"],
            "surprise": ["sad", "neutral"],
        }

        for i in range(1, len(self.frame_analyses)):
            prev = self.frame_analyses[i - 1]
            curr = self.frame_analyses[i]

            # Pega emoção dominante de cada frame
            prev_emotion = prev.emotions[0].emotion if prev.emotions else None
            curr_emotion = curr.emotions[0].emotion if curr.emotions else None

            if prev_emotion and curr_emotion and prev_emotion != curr_emotion:
                # Verifica se é uma mudança contrastante
                if curr_emotion in contrasting.get(prev_emotion, []):
                    prev_label = prev.emotions[0].emotion_label
                    curr_label = curr.emotions[0].emotion_label
                    anomalies.append({
                        "tipo": "Mudança de emoção",
                        "frame": curr.frame_number,
                        "descricao": f"De {prev_label} para {curr_label}"
                    })

        return anomalies

    def get_activity_anomalies(self) -> list[dict]:
        """
        Detecta anomalias de atividades.
        Anomalias incluem: movimentos bruscos e poses atípicas.
        """
        anomalies = []

        for fa in self.frame_analyses:
            for activity in fa.activities:
                if activity.is_anomaly:
                    anomalies.append({
                        "tipo": "Movimento anômalo",
                        "frame": activity.frame_number,
                        "descricao": activity.anomaly_type
                    })

        return anomalies

    def get_summary(self) -> dict:
        """Retorna um resumo da análise."""
        face_anomalies = self.get_face_anomalies()
        emotion_anomalies = self.get_emotion_anomalies()
        activity_anomalies = self.get_activity_anomalies()
        emotion_counts = self.get_emotion_counts()
        activity_counts = self.get_activity_counts()

        return {
            "video_path": self.video_path,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "fps": self.fps,
            "duration_seconds": round(self.duration_seconds, 2),
            # Seção 1: Reconhecimento Facial
            "total_face_detections": self.total_faces_detected,
            "frames_with_faces": self.frames_with_faces,
            "face_anomalies": face_anomalies,
            # Seção 2: Análise de Emoções
            "total_emotion_detections": self.total_emotions_detected,
            "emotion_counts": emotion_counts,
            "emotion_anomalies": emotion_anomalies,
            # Seção 3: Detecção de Atividades
            "total_activity_detections": self.total_activities_detected,
            "activity_counts": activity_counts,
            "activity_anomalies": activity_anomalies,
        }
