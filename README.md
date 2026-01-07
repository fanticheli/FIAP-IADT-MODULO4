# Video Analysis - Tech Challenge FIAP Fase 4

Aplicacao para analise de video com deteccao facial, analise de emocoes e deteccao de atividades.

## Funcionalidades

- **Reconhecimento Facial** - Deteccao de rostos usando MediaPipe
- **Analise de Emocoes** - Classificacao de expressoes usando DeepFace (7 emocoes)
- **Deteccao de Atividades** - Identificacao de posturas usando MediaPipe Pose
- **Deteccao de Anomalias** - Identifica mudancas bruscas em rostos, emocoes e atividades
- **Relatorio PDF** - Estatisticas organizadas por secao com prints de anomalias

## Requisitos

- Python 3.10+

## Instalacao

```bash
# 1. Clonar repositorio
git clone <url-do-repositorio>
cd FIAP-IADT-MODULO4

# 2. Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

## Uso

```bash
# Processar video padrao (videos/video.mp4)
python -m src.main

# Especificar video de entrada
python -m src.main -i meu_video.mp4

# Ver preview durante processamento
python -m src.main --preview

# Detectar rostos mais distantes
python -m src.main --confidence 0.3
```

## Saida

Os resultados sao salvos em `resultado/{nome_video}/`:
- `video_analyzed.mp4` - Video com rostos, emocoes e atividades marcados
- `relatorio.pdf` - Relatorio com estatisticas por secao
- `anomalia_frame_*.jpg` - Imagens dos frames com anomalias

## Estrutura do Projeto

```
src/
├── main.py                    # Ponto de entrada
├── config/settings.py         # Configuracoes
├── core/video_processor.py    # Orquestrador
├── analyzers/
│   ├── base_analyzer.py       # Interface
│   ├── face_analyzer_mp.py    # Deteccao facial (MediaPipe)
│   ├── emotion_analyzer.py    # Analise de emocoes (DeepFace)
│   └── activity_analyzer.py   # Deteccao de atividades (MediaPipe Pose)
├── models/detection.py        # Modelos de dados e anomalias
└── utils/
    ├── logger.py
    ├── visualization.py
    ├── report_generator.py
    └── frame_extractor.py
```

## Tecnologias

- **OpenCV** - Processamento de video
- **MediaPipe** - Deteccao facial e de pose
- **DeepFace** - Analise de emocoes
- **FPDF2** - Geracao de PDF

## Emocoes Detectadas

| Emocao | Ingles |
|--------|--------|
| Feliz | happy |
| Triste | sad |
| Raiva | angry |
| Surpresa | surprise |
| Medo | fear |
| Nojo | disgust |
| Neutro | neutral |

## Atividades Detectadas

| Atividade | Ingles |
|-----------|--------|
| Em pe | standing |
| Sentado | sitting |
| Bracos levantados | arms_raised |
| Inclinado | leaning |

## Anomalias Detectadas

| Tipo | Descricao |
|------|-----------|
| Mudanca brusca (facial) | Diferenca >= 2 rostos entre frames |
| Mudanca de emocao | Emocoes contrastantes (ex: feliz -> raiva) |
| Movimento brusco | Deslocamento rapido dos pontos corporais entre frames |
| Pose atipica | Angulos corporais extremos (tronco inclinado, membros em posicoes incomuns) |
