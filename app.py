import asyncio
import base64
from fastapi import FastAPI, WebSocket
from transformers import VitsModel, AutoTokenizer
import torch
import numpy as np
from scipy.io.wavfile import write
import IPython.display as ipd
import base64
import logging
import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from transformers import VitsModel, AutoTokenizer
import soundfile as sf

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tts_server")

# Cargar modelo una vez
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VitsModel.from_pretrained("facebook/mms-tts-spa").to(device).eval()
print("antes de ", model.config.sampling_rate)
model.config.sampling_rate = 17000  # Cambiar el sample rate del modelo
MODEL_SAMPLE_RATE = model.config.sampling_rate  # Actualizar el sample rate basado en la configuración del modelo
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-spa")

MODEL_SAMPLE_RATE = 22050  # confirmar que es el sample rate real del modelo

def generate_sine_wave(freq=440.0, duration=1.0, sample_rate=22050):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = 0.5 * np.sin(2 * np.pi * freq * t)  # amplitud 0.5
    pcm16 = (waveform * 32767).astype(np.int16)
    return pcm16.tobytes(), sample_rate

def synthesize_chunk(text_chunk: str) -> (bytes, int):
    inputs = tokenizer(text_chunk, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model(**inputs).waveform  # Tensor [1, T]
    waveform = output.squeeze().cpu().numpy()  # float32, idealmente en [-1,1] o cercano

    #sf.write('output.wav', output.squeeze().numpy(), model.config.sampling_rate)

    print(MODEL_SAMPLE_RATE, model.config.sampling_rate)

    # Logging rápido de diagnóstico
    peak = np.max(np.abs(waveform))
    logger.debug(f"Chunk '{text_chunk[:20]}...' peak before norm: {peak:.4f}, length samples: {waveform.shape[0]}")

    # Normalización solo si excede 1.0
    if peak > 1.0:
        waveform = waveform / peak
        logger.debug("Se aplicó normalización porque el peak era >1.0")

    # Recortar a [-1,1] por seguridad
    waveform = np.clip(waveform, -1.0, 1.0)

    # Si quisieras suavizar bordes entre chunks, podrías aplicar cross-fade o fade aquí.
    # Pero aplicar fade a cada chunk individual puede causar artefactos; se deja deshabilitado por defecto.
    # waveform = apply_fade(waveform, fade_len=512)

    # Convertir a PCM16
    pcm16 = (waveform * 32767).astype(np.int16)
    
    # sf.write('output2.wav', pcm16, model.config.sampling_rate)
    return pcm16.tobytes(), model.config.sampling_rate

def chunk_text(text: str, max_chars: int = 150):
    chunks = [sentence.strip() + "." for sentence in text.split(".") if sentence.strip()]
    final_chunks = []
    for chunk in chunks:
        sub_chunks = [sub_chunk.strip() + "," for sub_chunk in chunk.split(",") if sub_chunk.strip()]
        final_chunks.extend(sub_chunks if sub_chunks else [chunk])
    return final_chunks

@app.websocket("/ws/tts")
async def websocket_tts(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_json()
            # Soporte para tono de prueba
            if data.get("test_tone"):
                pcm_bytes, sample_rate = generate_sine_wave()
                sf.write('output2.wav', pcm_bytes, model.config.sampling_rate)
                
                frame_samples = int(0.1 * sample_rate)  # 100ms
                frame_size = frame_samples * 2  # 2 bytes por sample (PCM16)
                for i in range(0, len(pcm_bytes), frame_size):
                    frame = pcm_bytes[i : i + frame_size]
                    await ws.send_json({
                        "chunk": base64.b64encode(frame).decode("ascii"),
                        "sample_rate": sample_rate,
                        "format": "pcm16",
                        "last": i + frame_size >= len(pcm_bytes)
                    })
                await ws.send_json({"done": True})
                continue

            text = data.get("text", "")
            if not text:
                continue
            chunks = chunk_text(text)
            logger.info(f"Recibido texto para sintetizar: '{text[:30]}...' dividido en {len(chunks)} chunk(s)")

            for chunk in chunks:
                pcm_bytes, sample_rate = synthesize_chunk(chunk)
                # dividir en frames de ~100ms para envío incremental
                frame_samples = int(0.1 * sample_rate)  # 100 ms
                frame_size = frame_samples * 2  # 16-bit = 2 bytes
                for i in range(0, len(pcm_bytes), frame_size):
                    frame = pcm_bytes[i : i + frame_size]
                    await ws.send_json({
                        "chunk": base64.b64encode(frame).decode("ascii"),
                        "sample_rate": sample_rate,
                        "format": "pcm16",
                        "last": i + frame_size >= len(pcm_bytes)
                    })
            await ws.send_json({"done": True})
    except Exception as e:
        logger.exception(f"Error en WebSocket TTS: {e}")
        try:
            await ws.close()
        except:
            pass
