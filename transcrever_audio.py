

import os
import json
import datetime
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  CONFIGURAÇÕES — edite aqui
# ─────────────────────────────────────────────

AUDIO_FILE    = "exemplo.mp3"  # seu arquivo de áudio (tem que estar na mesma pasta)
WHISPER_MODEL = "small"                              # tiny | base | small | medium | large
LANGUAGE      = "pt"                               # pt = português | en = inglês
OUTPUT_TXT    = "transcricao.txt"
OUTPUT_JSON   = "transcricao.json"

# Diarização com Resemblyzer + sklearn
# Se você souber o número de participantes, coloque um inteiro: 2, 3, 4...
# Se deixar None, o script tenta estimar automaticamente.
NUM_SPEAKERS = None
MIN_SEGMENT_SECONDS = 0.75       # segmentos menores são expandidos para melhorar o embedding
MAX_AUTO_SPEAKERS = 8            # limite usado quando NUM_SPEAKERS = None

# ─────────────────────────────────────────────


def formatar_tempo(segundos: float) -> str:
    td = datetime.timedelta(seconds=round(segundos))
    total = int(td.total_seconds())
    h, resto = divmod(total, 3600)
    m, s = divmod(resto, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcrever(audio_path: str) -> list[dict]:
    """Transcreve o áudio com Whisper e retorna segmentos com timestamps."""
    import whisper

    print(f"[1/3] Carregando modelo Whisper '{WHISPER_MODEL}'...")
    modelo = whisper.load_model(WHISPER_MODEL)

    print("[2/3] Transcrevendo... (pode demorar alguns minutos)")
    resultado = modelo.transcribe(
        audio_path,
        language=LANGUAGE,
        verbose=False,
        word_timestamps=False,
    )
    return resultado["segments"]


def _recortar_wav(wav, inicio: float, fim: float, sr: int = 16000):
    """Recorta o array de áudio pré-processado pelo Resemblyzer."""
    import numpy as np

    inicio_i = max(0, int(inicio * sr))
    fim_i = min(len(wav), int(fim * sr))
    trecho = wav[inicio_i:fim_i]

    # Evita falha em segmentos vazios ou curtos demais.
    minimo = max(1, int(MIN_SEGMENT_SECONDS * sr))
    if len(trecho) == 0:
        return np.zeros(minimo, dtype=np.float32)
    if len(trecho) < minimo:
        trecho = np.pad(trecho, (0, minimo - len(trecho)))

    return trecho


def _estimar_num_falantes(embeddings, max_speakers: int) -> int:
    """
    Estima o número de falantes usando silhouette score.
    Para poucos segmentos, usa 1 ou 2 clusters de forma conservadora.
    """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    n = len(embeddings)
    if n <= 1:
        return 1
    if n == 2:
        return 2

    melhor_k = 2
    melhor_score = -1.0
    limite = min(max_speakers, n - 1)

    for k in range(2, limite + 1):
        modelo = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        labels = modelo.fit_predict(embeddings)

        # Silhouette exige pelo menos 2 labels e menos labels que amostras.
        if len(set(labels)) < 2 or len(set(labels)) >= n:
            continue

        score = silhouette_score(embeddings, labels, metric="cosine")
        if score > melhor_score:
            melhor_score = score
            melhor_k = k

    return melhor_k


def diarizar(segmentos: list[dict], audio_path: str) -> list[dict]:
    """
    Atribui um falante a cada segmento do Whisper usando:
      1. Resemblyzer para gerar embeddings de voz por segmento;
      2. scikit-learn para agrupar embeddings parecidos;
      3. rótulos SPEAKER_01, SPEAKER_02 etc.

    Essa abordagem é mais leve que pyannote e evita token/modelos do HuggingFace.
    """
    import numpy as np
    from resemblyzer import VoiceEncoder, preprocess_wav
    from sklearn.cluster import AgglomerativeClustering

    print("[2b/3] Executando diarização com Resemblyzer + sklearn...")

    if not segmentos:
        return []

    wav = preprocess_wav(audio_path)  # mono, 16 kHz, normalizado
    encoder = VoiceEncoder()

    embeddings = []
    segmentos_validos = []

    for seg in segmentos:
        inicio = float(seg["start"])
        fim = float(seg["end"])
        texto = seg.get("text", "").strip()

        if not texto:
            continue

        # Expande um pouco segmentos muito curtos, mantendo o centro original.
        duracao = fim - inicio
        if duracao < MIN_SEGMENT_SECONDS:
            centro = (inicio + fim) / 2
            inicio = max(0.0, centro - MIN_SEGMENT_SECONDS / 2)
            fim = min(len(wav) / 16000, centro + MIN_SEGMENT_SECONDS / 2)

        trecho = _recortar_wav(wav, inicio, fim)
        embedding = encoder.embed_utterance(trecho)

        embeddings.append(embedding)
        segmentos_validos.append(seg)

    if not embeddings:
        return []

    embeddings = np.asarray(embeddings)

    if NUM_SPEAKERS is None:
        n_speakers = _estimar_num_falantes(embeddings, MAX_AUTO_SPEAKERS)
        print(f"     Número estimado de falantes: {n_speakers}")
    else:
        n_speakers = max(1, min(int(NUM_SPEAKERS), len(embeddings)))
        print(f"     Número de falantes configurado: {n_speakers}")

    if n_speakers == 1:
        labels = np.zeros(len(embeddings), dtype=int)
    else:
        clusterizador = AgglomerativeClustering(
            n_clusters=n_speakers,
            metric="cosine",
            linkage="average",
        )
        labels = clusterizador.fit_predict(embeddings)

    # Dá nomes estáveis aos falantes pela primeira aparição no áudio.
    mapa_falantes: dict[int, str] = {}
    proximo = 1
    resultado = []

    for seg, label in zip(segmentos_validos, labels):
        label = int(label)
        if label not in mapa_falantes:
            mapa_falantes[label] = f"SPEAKER_{proximo:02d}"
            proximo += 1

        resultado.append({
            "inicio": float(seg["start"]),
            "fim": float(seg["end"]),
            "falante": mapa_falantes[label],
            "texto": seg.get("text", "").strip(),
        })

    return resultado


def salvar_txt(segmentos: list[dict], caminho: str) -> None:
    linhas = []
    falante_anterior = None

    for seg in segmentos:
        falante = seg["falante"]
        tempo = formatar_tempo(seg["inicio"])
        texto = seg["texto"]

        if falante != falante_anterior:
            linhas.append(f"\n[{falante}]")
            falante_anterior = falante

        linhas.append(f"  {tempo}  {texto}")

    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas).strip())

    print(f"\n✅  Transcrição salva em: {caminho}")


def salvar_json(segmentos: list[dict], caminho: str) -> None:
    dados = [
        {
            "inicio": formatar_tempo(s["inicio"]),
            "fim": formatar_tempo(s["fim"]),
            "falante": s["falante"],
            "texto": s["texto"],
        }
        for s in segmentos
    ]
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    print(f"✅  JSON salvo em: {caminho}")


def imprimir_resumo(segmentos: list[dict]) -> None:
    from collections import defaultdict

    tempos: dict[str, float] = defaultdict(float)
    for s in segmentos:
        tempos[s["falante"]] += s["fim"] - s["inicio"]

    print("\n── Resumo por participante ──────────────────")
    for falante, duracao in sorted(tempos.items()):
        print(f"  {falante:20s}  {formatar_tempo(duracao)}")
    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
#  EXECUÇÃO PRINCIPAL
# ─────────────────────────────────────────────

if __name__ == "__main__":
    if not os.path.exists(AUDIO_FILE):
        raise FileNotFoundError(f"Arquivo não encontrado: {AUDIO_FILE}")

    segmentos_whisper = transcrever(AUDIO_FILE)
    segmentos_finais = diarizar(segmentos_whisper, AUDIO_FILE)

    salvar_txt(segmentos_finais, OUTPUT_TXT)
    salvar_json(segmentos_finais, OUTPUT_JSON)
    imprimir_resumo(segmentos_finais)

    # Prévia no terminal
    print("── Prévia da transcrição ────────────────────")
    falante_ant = None
    for seg in segmentos_finais[:30]:
        if seg["falante"] != falante_ant:
            print(f"\n[{seg['falante']}]")
            falante_ant = seg["falante"]
        print(f"  {formatar_tempo(seg['inicio'])}  {seg['texto']}")
    print("\n(... veja o arquivo completo em transcricao.txt)")
