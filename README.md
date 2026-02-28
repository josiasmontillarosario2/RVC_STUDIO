# 🎙️ RVC Studio

RVC Studio es una aplicación basada en Next.js + RVC (Retrieval-based Voice Conversion) que permite ejecutar modelos de conversión de voz localmente.

Este proyecto requiere descargar modelos base manualmente debido a su tamaño.

---

# 📦 1) Descargar los Modelos Base (OBLIGATORIO)

Los modelos NO están incluidos en el repositorio.

Descárgalos desde Google Drive:

👉 https://drive.google.com/file/d/1H1_ddP26IGvlpLAHd4YbGCrjZbh0m6wz/view?usp=drive_link

### Pasos:

1. Descarga el archivo ZIP.
2. Descomprímelo.
3. Dentro encontrarás estas carpetas:
hubert/
rmvpe/
pretrained_v2/
weights/


4. Copia esas carpetas dentro de:
rvc_minimal/assets/




⚠️ Si las carpetas no están en esa ruta exacta, el sistema no funcionará.

---

# 🐍 2) Instalar Dependencias de Python

Se recomienda usar un entorno virtual o conda.

Desde la raíz del proyecto:


pip install -r requirements.txt
pip install -r rvc_minimal/requirements-api.txt




Si usas conda:

conda env create -f rvc_environment.yml
conda activate rvc




📦 3) Instalar Dependencias de Node (Next.js)

Desde la raíz del proyecto:

npm install
🚀 4) Ejecutar el Proyecto
npm run dev

Luego abre en tu navegador:

http://localhost:3000





⚙️ Requisitos

Node.js 18+

Python 3.10 recomendado

Conda (opcional pero recomendado)

GPU NVIDIA + CUDA (opcional pero recomendado para mejor rendimiento)

❗ Notas Importantes

Los modelos no se suben a GitHub debido a su tamaño.

Asegúrate de tener instalada la versión correcta de PyTorch (CPU o CUDA).

Si tienes problemas con CUDA, verifica tu instalación con:

python -c "import torch; print(torch.version.cuda)"
🧠 Estructura del Proyecto
rvc_minimal/
  assets/               # Modelos base (NO incluidos en repo)
  requirements-api.txt
scripts/
requirements.txt
package.json
🛠 Troubleshooting
Error: Torch / CUDA mismatch

Instala la versión correcta de PyTorch según tu GPU.

Error: ffmpeg no encontrado

Instala ffmpeg y agrégalo al PATH.

📄 Licencia

Uso educativo y experimental.


-------------------------------------------------------------------------------------------


🎧 Segmentación de Audio (Recomendado Antes de Entrenar)

Si tienes un audio largo (10, 15, 20 minutos o más), es altamente recomendable segmentarlo antes de entrenar para obtener mejores resultados.

Los archivos largos pueden:

Reducir la eficiencia del entrenamiento

Hacer inestable la extracción de F0

Aumentar el uso de memoria

Reducir la consistencia de la voz

Para un mejor rendimiento en RVC, divide las grabaciones largas en segmentos de 45 segundos.

🖥 Cómo Segmentar tu Audio

Este proyecto incluye una herramienta con interfaz gráfica en Python:

segmenter_ui.py
Pasos:

Ejecuta el programa:

python segmenter_ui.py

Selecciona el audio completo
Ejemplo:

voz18min.wav

Elige la carpeta donde se guardarán los audios segmentados.

Haz clic en Segmentar.

📂 Ejemplo de Resultado

Si tu archivo original es:

voz18min.wav

El programa generará:

voz18min_000.wav
voz18min_001.wav
voz18min_002.wav
...

Cada archivo:

Tendrá una duración de 45 segundos

Estará convertido a 40kHz

Será mono (1 canal)

Será WAV 16-bit PCM

Estará listo para entrenar en RVC

✅ ¿Por Qué Mejora los Resultados?

Segmentar audios largos:

Mejora la variación del dataset

Hace más estable la extracción de tono (F0)

Reduce el overfitting

Produce modelos de voz más limpios y consistentes

Para entrenar voz hablada (10–30 minutos en total), la segmentación es altamente recomendable.
