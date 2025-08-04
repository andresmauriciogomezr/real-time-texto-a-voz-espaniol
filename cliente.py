import streamlit as st
import asyncio
import websockets
import json
import base64
import wave
import io
from typing import Optional
import numpy as np
import streamlit.components.v1 as components

st.set_page_config(page_title="Realtime TTS Client", layout="centered")

st.title("Realtime TTS Client (Streamlit)")
st.markdown(
    """
    Escribe un texto y presiona **"Solicitar"**.  
    También puedes pedir un tono de prueba (440 Hz) para verificar pitch/velocidad.  
    El backend emite chunks PCM16 en base64 y este cliente permite:
    - Diagnóstico (duración/freq dominante)  
    - Reproducción incremental en el navegador vía Web Audio API.  
    """
)

backend_url = st.text_input("WebSocket endpoint", value="ws://localhost:8000/ws/tts")
mode = st.radio("Modo", ["Texto", "Tono de prueba (440Hz)"])
text = st.text_area("Texto a sintetizar", value="Qué tal funciona esta vuelta", height=120)
synthesize = st.button("Solicitar")

def build_wav(pcm_bytes: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()

def inspect_wav(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), 'rb') as wf:
        sr = wf.getframerate()
        frames = wf.getnframes()
        duration = frames / sr
        return {
            "sample_rate_header": sr,
            "frames": frames,
            "duration_sec": duration
        }

def estimate_dominant_freq(pcm_bytes: bytes, sample_rate: int):
    data = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    if len(data) == 0:
        return None
    hann = np.hanning(len(data))
    windowed = data * hann
    fft = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(len(windowed), d=1.0 / sample_rate)
    magnitude = np.abs(fft)
    peak_idx = np.argmax(magnitude)
    return freqs[peak_idx]

async def fetch_tts_audio(endpoint: str, text: str, is_test_tone: bool, progress_callback=None):
    pcm_accum = bytearray()
    sample_rate: Optional[int] = None
    try:
        async with websockets.connect(endpoint) as ws:
            payload = {"test_tone": True} if is_test_tone else {"text": text}
            await ws.send(json.dumps(payload))
            while True:
                msg = await ws.recv()
                try:
                    data = json.loads(msg)
                except Exception:
                    continue
                if data.get("done"):
                    break
                if "chunk" in data:
                    chunk_bytes = base64.b64decode(data["chunk"])
                    pcm_accum.extend(chunk_bytes)
                    if progress_callback:
                        progress_callback(len(pcm_accum))
                    if sample_rate is None and data.get("sample_rate"):
                        sample_rate = data.get("sample_rate")
    except Exception as e:
        raise RuntimeError(f"Error al conectar o recibir audio: {e}")
    if sample_rate is None:
        sample_rate = 22050
    return bytes(pcm_accum), sample_rate

if synthesize:
    if mode == "Texto" and not text.strip():
        st.warning("Escribe algo antes de sintetizar.")
    else:
        status = st.empty()
        progress_bar = st.progress(0)
        chunk_info = st.empty()

        status.info("Solicitando audio para diagnóstico...")

        try:
            def progress_cb(bytes_received):
                kb = bytes_received / 1024
                progress_bar.progress(min(1.0, kb / 200))
                chunk_info.markdown(f"Audio acumulado: {kb:0.1f} KB")

            is_test_tone = mode != "Texto"
            pcm_bytes, sample_rate = asyncio.run(
                fetch_tts_audio(backend_url, text, is_test_tone, progress_callback=progress_cb)
            )
            status.success("Audio recibido. Construyendo WAV de referencia...")
            wav_bytes = build_wav(pcm_bytes, sample_rate)

            # Diagnóstico
            info = inspect_wav(wav_bytes)
            st.subheader("Diagnóstico del audio recibido")
            st.markdown(f"- Sample rate en header: **{info['sample_rate_header']} Hz**")
            st.markdown(f"- Frames: **{info['frames']}**")
            st.markdown(f"- Duración (segundos): **{info['duration_sec']:.3f} s**")
            dominant = estimate_dominant_freq(pcm_bytes, sample_rate)
            if dominant:
                st.markdown(f"- Frecuencia dominante estimada: **{dominant:.1f} Hz**")
            else:
                st.markdown(f"- Frecuencia dominante: no disponible (audio vacío)")

            st.audio(wav_bytes, format="audio/wav", start_time=0)
            st.download_button("Descargar WAV", data=wav_bytes, file_name="tts.wav", mime="audio/wav")

            # Componente HTML/JS para reproducción por chunks en tiempo real
            st.subheader("Reproducción incremental (Web Audio API)")
            js = f"""
            <div>
              <p>Reproduciendo en tiempo real desde WebSocket: <code>{backend_url}</code> (modo: {mode})</p>
              <div id="status">Conectando...</div>
              <script>
                const sampleRateDisplay = document.createElement('div');
                const statusEl = document.getElementById('status');
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                let queue = [];
                let isPlaying = false;
                let targetSampleRate = null;
                let ws = new WebSocket("{backend_url.replace('ws://', 'ws://')}");
                ws.binaryType = 'arraybuffer';
                ws.onopen = () => {{
                  statusEl.textContent = 'WebSocket abierto';
                  const payload = {json.dumps({"test_tone": True} if is_test_tone else {"text": text})};
                  ws.send(JSON.stringify(payload));
                }};
                ws.onmessage = async (evt) => {{
                  try {{
                    const data = JSON.parse(evt.data);
                    if (data.sample_rate && !targetSampleRate) {{
                      targetSampleRate = data.sample_rate;
                      sampleRateDisplay.textContent = 'Sample rate recibido: ' + targetSampleRate + ' Hz';
                      document.body.appendChild(sampleRateDisplay);
                    }}
                    if (data.chunk) {{
                      const b64 = data.chunk;
                      const raw = atob(b64);
                      const buf = new ArrayBuffer(raw.length);
                      const view = new Uint8Array(buf);
                      for (let i = 0; i < raw.length; ++i) {{
                        view[i] = raw.charCodeAt(i);
                      }}
                      // PCM16 little endian to Float32
                      const int16 = new Int16Array(view.buffer);
                      const float32 = new Float32Array(int16.length);
                      for (let i = 0; i < int16.length; ++i) {{
                        float32[i] = int16[i] / 32767;
                      }}
                      queue.push(float32);
                      if (!isPlaying) playQueue();
                    }}
                    if (data.done) {{
                      statusEl.textContent = 'Síntesis completa';
                    }}
                  }} catch (e) {{
                    console.warn('error procesando mensaje', e);
                  }}
                }};

                function playQueue() {{
                  if (queue.length === 0) {{
                    isPlaying = false;
                    return;
                  }}
                  isPlaying = true;
                  const chunk = queue.shift();
                  const buffer = audioCtx.createBuffer(1, chunk.length, targetSampleRate || 22050);
                  buffer.getChannelData(0).set(chunk);
                  const src = audioCtx.createBufferSource();
                  src.buffer = buffer;
                  src.connect(audioCtx.destination);
                  src.onended = () => {{
                    playQueue();
                  }};
                  src.start();
                }}
              </script>
            </div>
            """
            components.html(js, height=200)
        except Exception as e:
            status.error(f"Falló la síntesis: {e}")
            st.stop()
