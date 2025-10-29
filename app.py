# -*- coding: utf-8 -*-
import os
import uuid
import glob
import json
import sys
import traceback
import logging
import threading
import time
import random
from datetime import datetime
from decimal import Decimal, InvalidOperation
from dotenv import load_dotenv
from typing import List, Any, Dict, Tuple, Optional

# --- INICIO: crear archivo temporal desde secret GCP_SERVICE_KEY (SOLO UNA VEZ) ---
_key = os.environ.get("GCP_SERVICE_KEY")
if _key:
    key_path = "/tmp/vision-key.json"
    with open(key_path, "w", encoding="utf-8") as f:
        f.write(_key)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path
# --- FIN ---

# ---------- Dependencias opcionales (import guardado) ----------
try:
    import gradio as gr
    from gradio.components.chatbot import ChatMessage
except Exception:
    gr = None
    ChatMessage = None
    print("⚠️ gradio no está instalado. Instálalo con: pip install gradio")

# PDF / DOCX
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None

# Images and pdf->image
try:
    from PIL import Image
except Exception:
    Image = None

try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None

# Local OCR fallback
try:
    import pytesseract
except Exception:
    pytesseract = None

# Google Vision (preferred OCR) - INICIALIZAR UNA SOLA VEZ
VISION_AVAILABLE = False
vision_client = None
try:
    from google.cloud import vision
    VISION_AVAILABLE = True
    if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        try:
            vision_client = vision.ImageAnnotatorClient()
            print("✅ Google Vision cliente inicializado (OCR preferente).")
        except Exception as e:
            vision_client = None
            print("⚠️ No se pudo inicializar Google Vision client:", e)
except Exception:
    vision = None
    VISION_AVAILABLE = False
    print("ℹ️ Google Vision librería no disponible; se usará OCR local si está instalado.")

# ---------- LangChain / Google GenAI (import guardado y diagnóstico) ----------
LANGCHAIN_IMPORTS_OK = False
_langchain_import_error = None

def _print_langchain_diagnostic(exc):
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        print("⚠️ importlib.metadata no disponible para diagnóstico de versiones.")
        print("Exception:", exc)
        return

    print("⚠️ Falló import LangChain/Google-GenAI:", exc)
    print("🔎 Diagnóstico versiones instaladas (intento):")
    pkgs = [
        "langchain", "langchain-core", "langchain-google-genai",
        "langchain_google_genai", "langchain-community", "langchain_community",
        "langchain_text_splitters", "chromadb", "pydantic"
    ]
    for p in pkgs:
        try:
            print(f"  - {p} => {version(p)}")
        except PackageNotFoundError:
            print(f"  - {p} => NOT INSTALLED")
        except Exception as e:
            print(f"  - {p} => error al consultar: {e}")

    print("\n✅ Sugerencias para resolver (ejecutar en el entorno / Dockerfile):")
    print("  pip install --no-cache-dir --upgrade langchain-core langchain langchain-google-genai langchain-community langchain_text_splitters chromadb")
    print("  Si sigue fallando, inspecciona pydantic (si está >2.0 puede haber incompatibilidades).")
    print("  Para forzar compatibilidad con paquetes que requieren pydantic v1:")
    print("  pip install 'pydantic<2.0.0'  # sólo si no rompe otras dependencias")
    print("  Añade las librerías a requirements.txt / Dockerfile para que HF las instale en build.\n")

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    # Los prompts / history moved between langchain versions -> intentamos rutas seguras
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except Exception:
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    try:
        from langchain_core.chat_history import BaseChatMessageHistory
    except Exception:
        from langchain.schema import BaseChatMessageHistory

    try:
        from langchain_community.chat_message_histories import ChatMessageHistory
    except Exception:
        ChatMessageHistory = None

    try:
        from langchain_core.runnables.history import RunnableWithMessageHistory
    except Exception:
        RunnableWithMessageHistory = None

    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
    except Exception:
        HarmCategory = None
        HarmBlockThreshold = None

    LANGCHAIN_IMPORTS_OK = True
    print("✅ LangChain / Google-GenAI imports OK")

except Exception as e:
    _langchain_import_error = e
    _print_langchain_diagnostic(e)
    LANGCHAIN_IMPORTS_OK = False


# ========== CORRECCIÓN: Embeddings Locales ==========
LOCAL_EMBEDDINGS_AVAILABLE = False
try:
    # CORRECTO: Importar directamente SentenceTransformer
    from sentence_transformers import SentenceTransformer
    LOCAL_EMBEDDINGS_AVAILABLE = True
    print("✅ SentenceTransformer disponible para embeddings locales")
except Exception as e:
    print(f"⚠️ SentenceTransformer no disponible: {e}")
    print("💡 Instala con: pip install sentence-transformers")

# Env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# ---------- LangChain / Google GenAI (import guardado y diagnóstico) ----------
LANGCHAIN_IMPORTS_OK = False
_langchain_import_error = None

def _print_langchain_diagnostic(exc):
    # imprime diagnóstico útil para los logs / HF build
    try:
        from importlib.metadata import version, PackageNotFoundError
    except Exception:
        print("⚠️ importlib.metadata no disponible para diagnóstico de versiones.")
        return

    print("⚠️ Falló import LangChain/Google-GenAI:", exc)
    print("🔎 Diagnóstico versiones instaladas (intento):")
    pkgs = [
        "langchain", "langchain-core", "langchain-google-genai",
        "langchain_google_genai", "langchain-community", "langchain_community",
        "langchain_text_splitters", "chromadb", "pydantic"
    ]
    for p in pkgs:
        try:
            print(f"  - {p} => {version(p)}")
        except PackageNotFoundError:
            print(f"  - {p} => NOT INSTALLED")
        except Exception as e:
            print(f"  - {p} => error al consultar: {e}")

    # sugerencias rápidas
    print("\n✅ Sugerencias para resolver (ejecutar en el entorno / Dockerfile):")
    print("  pip install --no-cache-dir --upgrade langchain-core langchain langchain-google-genai langchain-community langchain_text_splitters chromadb")
    print("  Si sigue fallando, inspecciona pydantic (si está >2.0 puede haber incompatibilidades).")
    print("  Para forzar compatibilidad con paquetes que requieren pydantic v1:")
    print("  pip install 'pydantic<2.0.0'  # sólo si no rompe otras dependencias")
    print("  Añade las librerías a requirements.txt / Dockerfile para que HF las instale en build.\n")

try:
    # intento original (tu lista de imports)
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.vectorstores import Chroma
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain.chains import create_history_aware_retriever, create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    # los prompts / history moved between langchain versions -> intentamos ambas rutas
    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    except Exception:
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

    try:
        from langchain_core.chat_history import BaseChatMessageHistory
    except Exception:
        from langchain.schema import BaseChatMessageHistory

    try:
        from langchain_community.chat_message_histories import ChatMessageHistory
    except Exception:
        # en algunas versiones está en langchain.chat_models / o no existe la clase
        ChatMessageHistory = None

    try:
        from langchain_core.runnables.history import RunnableWithMessageHistory
    except Exception:
        # fallback: marcar None — la app hará checks y usará modo fallback si no está presente
        RunnableWithMessageHistory = None

    # Google generative ai types (si están)
    try:
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
    except Exception:
        HarmCategory = None
        HarmBlockThreshold = None

    LANGCHAIN_IMPORTS_OK = True
    print("✅ LangChain / Google-GenAI imports OK")

except Exception as e:
    _langchain_import_error = e
    _print_langchain_diagnostic(e)
    LANGCHAIN_IMPORTS_OK = False


# ========== CONFIGURACIÓN OPTIMIZADA ==========
# Modelo de Gemini (CORREGIDO)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")  # Cambiado de gemini-2.5-pro a nombre correcto
print(f"🤖 Modelo Gemini seleccionado: {GEMINI_MODEL}")

# Embeddings locales
USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "true").lower() == "true"
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LOCAL_EMBEDDING_DEVICE = os.getenv("LOCAL_EMBEDDING_DEVICE", "cpu")

# Configuración de fragmentación
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "3000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "300"))

# Modelo ligero para espacios limitados
USE_LIGHTWEIGHT_MODEL = os.getenv("USE_LIGHTWEIGHT_MODEL", "true").lower() == "true"

if USE_LIGHTWEIGHT_MODEL:
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB
    print("🪶 Modo ligero activado: modelo ~80MB")
else:
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # ~400MB
    print("💪 Modo completo: modelo multilingüe ~400MB")

# ---------- OCR helpers ----------
def ocr_image_with_google_bytes(image_bytes):
    if not vision_client:
        return None
    try:
        image = vision.Image(content=image_bytes)
        response = vision_client.text_detection(image=image)
        if getattr(response, "error", None) and getattr(response.error, "message", None):
            print("Google Vision error:", response.error.message)
            return None
        texts = response.text_annotations
        if texts:
            return texts[0].description
        return ""
    except Exception as e:
        print("Error Google Vision OCR:", e)
        return None

def ocr_image_pil_local(pil_img):
    if not pytesseract:
        return "[OCR local no disponible: pytesseract no instalado]"
    try:
        return pytesseract.image_to_string(pil_img, lang='spa')
    except Exception as e:
        return f"[OCR local error: {e}]"

# ---------- Extract text from files (robusta) ----------
def extract_text_from_files(files, max_chars=250000):
    if not files:
        return ""
    full_text = ""
    for file_obj in files:
        file_path = getattr(file_obj, "name", str(file_obj))
        file_name = os.path.basename(file_path)
        try:
            text_block = f"\n\n--- INICIO DEL DOCUMENTO: {file_name} ---\n"
            lower = file_path.lower()

            if lower.endswith(".pdf"):
                native_text = ""
                if pypdf:
                    try:
                        reader = pypdf.PdfReader(file_path)
                        pages_text = [page.extract_text() or "" for page in reader.pages]
                        native_text = "\n".join(pages_text).strip()
                    except Exception as e:
                        print(f"⚠️ pypdf fallo en {file_name}: {e}")

                if native_text:
                    text_block += native_text
                else:
                    text_block += "\n[No se detectó texto nativo en el PDF. Intentando OCR por páginas...]\n"
                    if convert_from_path and (vision_client or pytesseract):
                        try:
                            pil_pages = convert_from_path(file_path, dpi=200)
                            for i, pil in enumerate(pil_pages):
                                page_bytes = None
                                try:
                                    from io import BytesIO
                                    bio = BytesIO()
                                    pil.save(bio, format="PNG")
                                    page_bytes = bio.getvalue()
                                except Exception:
                                    page_bytes = None

                                if vision_client and page_bytes:
                                    g_text = ocr_image_with_google_bytes(page_bytes)
                                    if g_text is not None:
                                        text_block += f"\n[OCR página {i+1} - GoogleVision]\n" + g_text + "\n"
                                        continue

                                if pytesseract:
                                    text_block += f"\n[OCR página {i+1} - OCR local]\n" + ocr_image_pil_local(pil) + "\n"
                                else:
                                    text_block += f"\n[No hay OCR disponible para la página {i+1}]\n"
                        except Exception as e:
                            text_block += f"\n[Error convirtiendo PDF a imagenes para OCR: {e}]\n"
                    else:
                        text_block += "\n[No disponible: pdf2image/poppler o motor OCR (Vision/pytesseract) faltante]\n"

            elif lower.endswith(".docx"):
                if docx:
                    try:
                        doc = docx.Document(file_path)
                        text_block += "\n".join([para.text for para in doc.paragraphs])
                    except Exception as e:
                        text_block += f"\n[Error leyendo docx: {e}]\n"
                else:
                    text_block += "\n[python-docx no instalado]\n"

            elif lower.endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp")):
                used = False
                try:
                    with open(file_path, "rb") as f:
                        img_bytes = f.read()
                    if vision_client:
                        g_text = ocr_image_with_google_bytes(img_bytes)
                        if g_text is not None:
                            text_block += "[OCR - Google Vision]\n" + g_text + "\n"
                            used = True
                except Exception as e:
                    print("Error leyendo imagen para Vision:", e)
                if not used:
                    if Image and pytesseract:
                        try:
                            pil = Image.open(file_path)
                            text_block += "[OCR - local]\n" + ocr_image_pil_local(pil) + "\n"
                        except Exception as e:
                            text_block += f"\n[Error leyendo imagen: {e}]\n"
                    else:
                        text_block += "\n[No hay motor OCR disponible para imagen]\n"

            else:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text_block += f.read()
                except Exception as e:
                    text_block += f"\n[Error leyendo archivo de texto: {e}]\n"

            text_block += f"\n--- FIN DEL DOCUMENTO: {file_name} ---\n"

            if len(full_text) + len(text_block) > max_chars:
                remaining = max_chars - len(full_text)
                full_text += text_block[:remaining] + "\n[Truncado: documento demasiado largo]\n"
                break
            else:
                full_text += text_block
            print(f"✅ Texto extraído de '{file_name}'.")
        except Exception as e:
            err = f"\n[Error extrayendo texto de '{file_name}': {e}]\n"
            print("🚨", err)
            full_text += err
    return full_text

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
PDF_DIRECTORY = os.getenv('PDF_DIRECTORY', 'data')
rutas_documentos = [
    os.path.join(PDF_DIRECTORY, "ULTIMA ACTUALIZACION COMPLETA (Diciembre 2024).pdf"),
    os.path.join(PDF_DIRECTORY, "ba ven nif 0.pdf"),
    os.path.join(PDF_DIRECTORY, "ba ven nif 2.pdf"),
    os.path.join(PDF_DIRECTORY, "ba-ven-nif-4.pdf"),
    os.path.join(PDF_DIRECTORY, "ba-ven-nif-5.pdf"),
    os.path.join(PDF_DIRECTORY, "ba-ven-nif-6.pdf"),
    os.path.join(PDF_DIRECTORY, "ba-ven-nif-7.pdf"),
    os.path.join(PDF_DIRECTORY, "ba ven nif 8.pdf"),
    os.path.join(PDF_DIRECTORY, "BAVENNIF09V0.pdf"),
    os.path.join(PDF_DIRECTORY, "ba-ven-nif-10.pdf"),
    os.path.join(PDF_DIRECTORY, "BA-VEN-NIF-11.pdf"),
    os.path.join(PDF_DIRECTORY, "ba ven nif 12.pdf"),
]
rutas_documentos = os.getenv('RUTAS_DOCUMENTOS') and json.loads(os.getenv('RUTAS_DOCUMENTOS')) or rutas_documentos

try:
    all_pdfs = sorted(glob.glob(os.path.join(PDF_DIRECTORY, "*.pdf")))
    for p in all_pdfs:
        if p not in rutas_documentos:
            rutas_documentos.append(p)
except Exception:
    all_pdfs = []

# ------------------------------------------------------------------
# LOGGER
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
print("STARTING APP - logs iniciales saldrán aquí.")

# ------------------------------------------------------------------
# Compatibilidad mínima: fallback para gr.Box
# ------------------------------------------------------------------
def get_box(*args, **kwargs):
    if gr is None:
        raise RuntimeError("gradio no disponible: no se puede crear la UI")
    if hasattr(gr, 'Box'):
        return gr.Box(*args, **kwargs)
    for name in ('Card', 'Group', 'Column'):
        cls = getattr(gr, name, None)
        if cls is not None:
            return cls(*args, **kwargs)
    return gr.Column(*args, **kwargs)

# ----------------------------
# UTIL: Limpiar espacio en disco
# ----------------------------
def cleanup_temp_files():
    """Limpia archivos temporales para liberar espacio"""
    try:
        import shutil
        temp_dirs = ['/tmp', os.path.expanduser('~/.cache')]
        cleaned = 0
        
        for temp_dir in temp_dirs:
            if os.path.exists(temp_dir):
                for item in os.listdir(temp_dir):
                    if item.startswith(('gradio', 'tmp', 'temp')):
                        try:
                            item_path = os.path.join(temp_dir, item)
                            if os.path.isfile(item_path):
                                os.unlink(item_path)
                                cleaned += 1
                            elif os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                                cleaned += 1
                        except Exception:
                            pass
        
        logger.info(f"🧹 Limpiados {cleaned} archivos temporales")
        return cleaned
    except Exception as e:
        logger.warning(f"⚠️ Error limpiando archivos temporales: {e}")
        return 0

# ----------------------------
# UTIL: escape de llaves
# ----------------------------
def _escape_braces_for_system(prompt):
    if prompt is None:
        return ""
    if not isinstance(prompt, str):
        prompt = str(prompt)
    placeholder = "___CONTEXT_PLACEHOLDER___"
    prompt = prompt.replace("{context}", placeholder)
    prompt = prompt.replace("{", "{{").replace("}", "}}")
    prompt = prompt.replace(placeholder, "{context}")
    return prompt

# ----------------------------
# GLOBALS RAG / STATE
# ----------------------------
todos_los_fragmentos: List[Any] = []
chroma_dir = os.getenv('CHROMA_DIR', './chroma_db_niif')
retriever = None
llm = None
conversational_rag_chain = None
history_aware_retriever = None

RAG_STATUS = {
    "initialized": False,
    "message": "not_started",
    "error": None,
    "progress": "0/0",
    "embeddings_created": False,
    "embedding_type": "unknown",
}

RAG_INIT_LOCK = threading.Lock()
RAG_INIT_EVENT = threading.Event()

# ------------------------------------------------------------------
# PROMPT MEJORADO: Con capacidades de razonamiento contable
# ------------------------------------------------------------------
QA_SYSTEM_PROMPT = (
    "Eres Jarvis, un Contador Público Autorizado (CPA) experto en NIIF (Normas Internacionales de Información Financiera) "
    "para Venezuela, con capacidades avanzadas de razonamiento y análisis contable.\n\n"
    
    "## TU PROCESO DE RAZONAMIENTO CONTABLE:\n\n"
    "Cuando analices una consulta contable, SIEMPRE sigue este proceso mental paso a paso:\n\n"
    
    "1. **ANÁLISIS INICIAL**: Identifica la transacción/evento y elementos financieros involucrados\n"
    "2. **IDENTIFICACIÓN NORMATIVA**: Determina qué NIIF/NIC/VEN-NIF específica aplica\n"
    "3. **RECONOCIMIENTO Y MEDICIÓN**: Aplica criterios técnicos de reconocimiento y valoración\n"
    "4. **REGISTRO CONTABLE**: Define débitos, créditos y cuentas del plan contable\n"
    "5. **PRESENTACIÓN Y REVELACIÓN**: Determina ubicación en EEFF y notas explicativas\n"
    "6. **VERIFICACIÓN**: Valida que Debe = Haber y consistencia con marco conceptual\n\n"
    
    "## CONTEXTO NORMATIVO DISPONIBLE:\n"
    "{context}\n\n"
    
    "## INSTRUCCIONES PRINCIPALES:\n\n"
    "1. **PRIORIZA las normas NIIF** del contexto normativo proporcionado arriba\n"
    "2. **MUESTRA TU RAZONAMIENTO**: Explica el 'por qué' de cada decisión contable, citando normas específicas\n"
    "3. **USA LENGUAJE PROFESIONAL PERO ACCESIBLE**: Técnico pero explicable\n"
    "4. Si el usuario sube documentos (facturas, contratos, EEFF):\n"
    "   - Extrae información clave (montos, fechas, conceptos)\n"
    "   - Aplica las normas NIIF relevantes del contexto\n"
    "   - Proporciona respuestas específicas basadas en ambos: normas + documentos usuario\n"
    "5. Para asientos contables:\n"
    "   - Debe(s) = Haber(es) siempre (validación aritmética)\n"
    "   - Usa cuentas conforme al plan contable NIIF\n"
    "   - Cita la norma aplicable (ej: NIC 2 párrafo 9, NIIF 15 párrafo 31)\n"
    "   - Incluye el concepto/glosa del asiento\n"
    "6. **FORMATO PREFERIDO PARA ANÁLISIS COMPLEJOS**:\n"
    "   - 🔍 ANÁLISIS DEL CASO\n"
    "   - 📚 NORMATIVA APLICABLE (con citas específicas)\n"
    "   - 💼 TRATAMIENTO CONTABLE (reconocimiento, medición)\n"
    "   - 📝 ASIENTOS CONTABLES (con tabla formateada)\n"
    "   - ⚠️ CONSIDERACIONES ESPECIALES (juicios, estimaciones, alertas)\n"
    "   - 📊 PRESENTACIÓN EN EEFF (ubicación y revelaciones)\n\n"
    
    "## CAPACIDADES DE RAZONAMIENTO:\n"
    "- Piensa paso a paso en problemas complejos\n"
    "- Considera alternativas y evalúa pros/contras\n"
    "- Identifica supuestos y sus implicaciones\n"
    "- Alerta sobre áreas grises o múltiples interpretaciones válidas\n\n"
    
    "Si no encuentras información suficiente en el contexto normativo o documentos del usuario, "
    "indícalo claramente y sugiere qué documentación adicional sería útil."
)

# ------------------------------------------------------------------
# AUDIT
# ------------------------------------------------------------------
AUDIT_LOG_PATH = os.getenv('AUDIT_LOG_PATH', 'audit.jsonl')

def audit_record(record: Dict[str, Any]):
    try:
        record['timestamp'] = datetime.utcnow().isoformat() + 'Z'
        with open(AUDIT_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        logger.exception('Fallo al registrar auditoría')

# ------------------------------------------------------------------
# VALIDATION: Journal JSON
# ------------------------------------------------------------------
def validate_journal_json(jjson: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if jjson.get('task') != 'journal_entry':
        return True, None
    try:
        entries = jjson['result']['entries']
    except KeyError:
        return False, {'error': 'invalid_schema', 'reason': 'missing entries'}
    total_debit = Decimal('0')
    total_credit = Decimal('0')
    for e in entries:
        try:
            d = Decimal(str(e.get('debit', 0)))
            c = Decimal(str(e.get('credit', 0)))
        except InvalidOperation:
            return False, {'error': 'invalid_number_format', 'entry': e}
        total_debit += d
        total_credit += c
    if total_debit != total_credit:
        return False, {'error': 'debits_not_equal_credits', 'delta': str(total_debit - total_credit)}
    return True, None

# ------------------------------------------------------------------
# LLM wrapper & helpers
# ------------------------------------------------------------------
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("DEFAULT_MAX_OUTPUT_TOKENS", "8192"))  # Aumentado para razonamiento

def call_llm_for_task(llm_client, prompt_str: str, task_type: str = 'explain', max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS):
    try:
        if llm_client is None:
            raise RuntimeError("LLM client no configurado.")
        response = llm_client.invoke(prompt_str)
        if hasattr(response, "content"):
            content = response.content
        elif isinstance(response, dict) and "content" in response:
            content = response["content"]
        else:
            content = str(response)
        logger.info(f"✅ LLM respondió: {len(content)} caracteres")
        return content
    except Exception:
        logger.exception('❌ Error llamando al LLM')
        raise

# ========== CORRECCIÓN CRÍTICA: Inicialización LLM con Gemini 2.5 Pro ==========
if LANGCHAIN_IMPORTS_OK and GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,  # Usar variable configurada ('gemini-2.5-pro' o 'gemini-2.5-pro-exp-03-25')
            temperature=0.3,  # Más bajo para precisión contable (antes 0.5)
            max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            }
        )
        logger.info(f"✅ LLM Gemini {GEMINI_MODEL} instanciado correctamente")
        logger.info(f"   Temperatura: 0.3 (precisión contable)")
        logger.info(f"   Max tokens: {DEFAULT_MAX_OUTPUT_TOKENS}")
    except Exception as e:
        logger.error(f"❌ Error instanciando LLM: {e}")
        llm = None
else:
    llm = None
    logger.warning("⚠️ LLM no inicializado: verifica LANGCHAIN_IMPORTS_OK y GOOGLE_API_KEY.")

# ========== CORRECCIÓN: Clase de Embeddings Locales Compatible ==========
class CustomLocalEmbeddings:
    """
    Wrapper personalizado para SentenceTransformer compatible con LangChain
    CORRIGE el error 'cache_folder' de HuggingFaceEmbeddings
    """
    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        
        logger.info(f"📥 Cargando modelo de embeddings: {model_name}")
        logger.info(f"   Device: {device}")
        
        try:
            # CORRECCIÓN: Llamada directa a SentenceTransformer SIN cache_folder
            self.model = SentenceTransformer(model_name, device=device)
            logger.info(f"✅ Total fragmentos extraídos: {len(todos_los_fragmentos)}")

            # 3. VERIFICAR ESPACIO EN DISCO
            try:
                import shutil
                total, used, free = shutil.disk_usage("/")
                free_gb = free // (2**30)
                logger.info(f"💾 Espacio disponible: {free_gb} GB")
                
                if free_gb < 1:
                    error_msg = (
                        f"❌ ESPACIO CRÍTICO: Solo {free_gb}GB libres\n"
                        "Se requiere al menos 1GB para crear embeddings.\n"
                        "La app continuará en modo LLM directo sin RAG."
                    )
                    logger.error(error_msg)
                    RAG_STATUS['error'] = error_msg
                    RAG_STATUS['message'] = "error_no_space"
                    return
            except Exception as disk_err:
                logger.warning(f"⚠️ No se pudo verificar espacio: {disk_err}")

            # 4. CREAR O CARGAR VECTOR STORE
            vector_store = None
            
            # OPCIÓN A: Cargar Vector Store existente (PRIORIDAD)
            if os.path.exists(chroma_dir) and os.listdir(chroma_dir):
                logger.info("♻️ Vector Store existente detectado. Intentando cargar...")
                logger.info(f"📂 Directorio: {chroma_dir}")
                
                try:
                    # Crear instancia de embeddings para cargar (no para generar)
                    if USE_LOCAL_EMBEDDINGS and LOCAL_EMBEDDINGS_AVAILABLE:
                        embeddings = CustomLocalEmbeddings(
                            model_name=LOCAL_EMBEDDING_MODEL,
                            device=LOCAL_EMBEDDING_DEVICE
                        )
                        RAG_STATUS['embedding_type'] = 'local'
                    else:
                        raise RuntimeError("Embeddings locales no disponibles")
                    
                    # CARGAR vector store (rápido, sin regenerar)
                    vector_store = Chroma(
                        persist_directory=chroma_dir,
                        embedding_function=embeddings
                    )
                    
                    # Verificar que contiene datos
                    try:
                        collection_count = vector_store._collection.count()
                        logger.info(f"✅ Vector Store cargado: {collection_count} vectores")
                        RAG_STATUS['embeddings_created'] = True
                        RAG_STATUS['message'] = f"✅ Cargado desde disco ({collection_count} vectores)"
                    except Exception:
                        # Si falla el count, asumir que está corrupto
                        logger.warning("⚠️ Vector Store parece corrupto")
                        vector_store = None
                    
                except Exception as e:
                    logger.warning(f"⚠️ No se pudo cargar Vector Store: {e}")
                    logger.warning("🔄 Se recreará desde cero...")
                    vector_store = None
            
            # OPCIÓN B: Crear nuevo Vector Store
            if vector_store is None and todos_los_fragmentos:
                logger.info("🆕 Creando nuevo Vector Store...")
                
                if USE_LOCAL_EMBEDDINGS and LOCAL_EMBEDDINGS_AVAILABLE:
                    RAG_STATUS['message'] = f"🧠 Creando embeddings para {len(todos_los_fragmentos)} fragmentos..."
                    
                    logger.info("="*80)
                    logger.info("🧠 CREACIÓN DE EMBEDDINGS LOCALES")
                    logger.info(f"   Modelo: {LOCAL_EMBEDDING_MODEL}")
                    logger.info(f"   Device: {LOCAL_EMBEDDING_DEVICE}")
                    logger.info(f"   Fragmentos: {len(todos_los_fragmentos)}")
                    logger.info(f"   Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
                    logger.info(f"   💰 GRATUITO - Sin límites de cuota")
                    if USE_LIGHTWEIGHT_MODEL:
                        logger.info(f"   🪶 Modelo ligero (~80MB)")
                        logger.info(f"   ⏱️ Tiempo estimado: 2-5 minutos")
                    else:
                        logger.info(f"   💪 Modelo completo (~400MB)")
                        logger.info(f"   ⏱️ Tiempo estimado: 5-10 minutos")
                    logger.info("="*80)
                    
                    try:
                        # CORRECCIÓN: Usar CustomLocalEmbeddings en lugar de HuggingFaceEmbeddings
                        embeddings = CustomLocalEmbeddings(
                            model_name=LOCAL_EMBEDDING_MODEL,
                            device=LOCAL_EMBEDDING_DEVICE
                        )
                        RAG_STATUS['embedding_type'] = 'local'
                        
                        logger.info("📦 Creando ChromaDB con embeddings locales...")
                        logger.info("   (Este proceso puede tardar varios minutos)")
                        
                        vector_store = Chroma.from_documents(
                            documents=todos_los_fragmentos,
                            embedding=embeddings,
                            persist_directory=chroma_dir
                        )
                        
                        logger.info("✅ Vector Store creado y persistido exitosamente")
                        logger.info(f"📂 Guardado en: {chroma_dir}")
                        
                        # Verificar tamaño
                        try:
                            total_size = sum(
                                os.path.getsize(os.path.join(dirpath, filename))
                                for dirpath, dirnames, filenames in os.walk(chroma_dir)
                                for filename in filenames
                            )
                            size_mb = total_size / (1024**2)
                            logger.info(f"📊 Tamaño Vector Store: {size_mb:.1f} MB")
                        except Exception:
                            pass
                        
                        RAG_STATUS['embeddings_created'] = True
                        
                    except Exception as e:
                        logger.exception(f"❌ Error creando embeddings locales: {e}")
                        RAG_STATUS['error'] = str(e)
                        RAG_STATUS['message'] = "error_creating_embeddings"
                        vector_store = None
                
                else:
                    error_msg = (
                        "❌ EMBEDDINGS LOCALES NO DISPONIBLES\n"
                        "Instala: pip install sentence-transformers torch"
                    )
                    logger.error(error_msg)
                    RAG_STATUS['error'] = error_msg
                    RAG_STATUS['message'] = "error_embeddings_unavailable"
                    vector_store = None

            # 5. CREAR RETRIEVER
            if vector_store:
                retriever = vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 6}  # Top 6 fragmentos más relevantes
                )
                logger.info("✅ Retriever configurado (k=6, similarity search)")
            else:
                retriever = None
                logger.warning("⚠️ No se creó retriever. Modo LLM directo sin RAG.")
                if not RAG_STATUS.get('error'):
                    RAG_STATUS['message'] = "fallback_mode_no_rag"

            # 6. CREAR CHAINS RAG (si hay retriever y LLM)
            if retriever and llm:
                logger.info("🔗 Creando cadenas RAG conversacionales...")
                
                # Prompt para contextualización (reformula pregunta considerando historial)
                contextualize_q_system_prompt = (
                    "Dado un historial de chat y la última pregunta del usuario "
                    "que podría referenciar el contexto del historial, "
                    "reformula la pregunta para que sea independiente y clara. "
                    "NO respondas la pregunta, solo reformúlala si es necesario."
                )
                contextualize_q_prompt = ChatPromptTemplate.from_messages([
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                # Prompt principal para Q&A
                qa_system_prompt = _escape_braces_for_system(QA_SYSTEM_PROMPT)
                qa_prompt = ChatPromptTemplate.from_messages([
                    ("system", qa_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ])

                # Retriever consciente del historial
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                
                # Chain para combinar documentos y responder
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
                
                # Chain de recuperación completa
                rag_chain = create_retrieval_chain(
                    history_aware_retriever,
                    question_answer_chain
                )

                # Message store para historial conversacional
                langchain_message_store = {}

                def get_session_history(session_id: str) -> BaseChatMessageHistory:
                    if session_id not in langchain_message_store:
                        langchain_message_store[session_id] = ChatMessageHistory()
                    return langchain_message_store[session_id]

                # Chain conversacional final
                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
                
                RAG_STATUS['initialized'] = True
                RAG_STATUS['message'] = "✅ Sistema RAG completo con embeddings locales"
                
                logger.info("="*80)
                logger.info("✅ INICIALIZACIÓN RAG COMPLETADA")
                logger.info(f"   Modelo LLM: {GEMINI_MODEL}")
                logger.info(f"   Embeddings: {RAG_STATUS['embedding_type']} ({LOCAL_EMBEDDING_MODEL.split('/')[-1]})")
                logger.info(f"   Fragmentos: {len(todos_los_fragmentos)}")
                logger.info(f"   Vector Store: {chroma_dir}")
                logger.info("   🎯 Sistema listo para consultas contables NIIF")
                logger.info("="*80)
            else:
                RAG_STATUS['initialized'] = False
                RAG_STATUS['message'] = "⚠️ Modo fallback: LLM directo sin contexto RAG"
                logger.warning("⚠️ Chains RAG no creadas. Usando LLM directo.")

        except Exception as e:
            RAG_STATUS['error'] = str(e)
            RAG_STATUS['message'] = "error_initialization"
            RAG_STATUS['initialized'] = False
            logger.exception("❌ EXCEPCIÓN CRÍTICA durante inicialización RAG")
        
        finally:
            RAG_INIT_EVENT.set()
            logger.info("🏁 Inicialización RAG finalizada")

# Iniciar thread de inicialización
if LANGCHAIN_IMPORTS_OK:
    init_thread = threading.Thread(target=initialize_rag_and_llm, daemon=True)
    init_thread.start()
    logger.info("🚀 Thread de inicialización RAG iniciado en segundo plano")
else:
    RAG_STATUS['message'] = "disabled_no_langchain"
    RAG_INIT_EVENT.set()

# ------------------------------------------------------------------
# Funciones UI / proceso
# ------------------------------------------------------------------
WELCOME_MESSAGE = [[
    None,
    "¡Hola! Soy **Jarvis**, tu asistente contable experto en NIIF.\n\n"
    "Estoy aquí para ayudarte con:\n"
    "- 📚 Consultas sobre normas NIIF/NIC/VEN-NIF\n"
    "- 📊 Análisis de documentos contables (facturas, contratos, EEFF)\n"
    "- 📝 Elaboración de asientos contables\n"
    "- 💡 Interpretación y aplicación de normas\n\n"
    "¿En qué puedo ayudarte hoy?"
]]

def get_rag_status():
    status = dict(RAG_STATUS)
    status['is_ready'] = RAG_INIT_EVENT.is_set()
    return status

def wait_for_rag_init(timeout=600):
    """Espera a que RAG termine de inicializar (máx timeout segundos)"""
    logger.info(f"⏳ Esperando inicialización RAG (timeout={timeout}s)...")
    ready = RAG_INIT_EVENT.wait(timeout=timeout)
    if ready:
        logger.info(f"✅ RAG init completado. Status: {RAG_STATUS['message']}")
    else:
        logger.warning(f"⚠️ Timeout esperando RAG init después de {timeout}s")
    return ready

# ------------------
# Chat handler MEJORADO
# ------------------
def _chat_fn(message, displayed_history, files, audio, current_chat_id, chats, filtered_order):
    """Handler principal de chat con RAG y manejo robusto"""
    
    # Validar sesión
    if not current_chat_id or current_chat_id not in chats:
        samples = [[chats[id_]["title"]] for id_ in filtered_order]
        yield displayed_history, "", chats, gr.Dataset(samples=samples), filtered_order
        return

    session_id = current_chat_id
    current_hist = chats[current_chat_id]["history"]
    user_message = str(message or "").strip()

    # Si no hay mensaje ni archivos, salir
    if not user_message and not files:
        yield current_hist, "", chats, gr.skip(), gr.skip()
        return

    # Agregar mensaje del usuario
    display_msg = user_message if user_message else "📎 [Documentos adjuntos]"
    current_hist.append([display_msg, None])
    yield current_hist, "", chats, gr.skip(), gr.skip()

    # Mostrar estado de procesamiento
    current_hist.append([None, "🔍 Procesando tu consulta..."])
    yield current_hist, "", chats, gr.skip(), gr.skip()

    # ESPERAR RAG si no está listo (primera consulta)
    if not RAG_INIT_EVENT.is_set():
        current_hist[-1][1] = (
            "⏳ **Inicializando sistema RAG por primera vez...**\n\n"
            "Esto puede tardar 5-10 minutos:\n"
            "- Descargando modelo de embeddings (~80-400MB)\n"
            "- Procesando documentos NIIF\n"
            "- Creando base de datos vectorial\n\n"
            "Responderé tan pronto esté listo. ⏱️"
        )
        yield current_hist, "", chats, gr.skip(), gr.skip()
        
        ready = wait_for_rag_init(timeout=600)  # 10 min
        
        if not ready:
            current_hist[-1][1] = (
                "⚠️ **Inicialización tardando más de lo esperado**\n\n"
                f"Estado: {RAG_STATUS['message']}\n\n"
                "Responderé con conocimiento base mientras se completa."
            )
            yield current_hist, "", chats, gr.skip(), gr.skip()
            time.sleep(2)

    # EXTRAER TEXTO DE ARCHIVOS DEL USUARIO
    docs_text = ""
    if files:
        try:
            current_hist[-1][1] = f"📎 Procesando {len(files)} archivo(s)..."
            yield current_hist, "", chats, gr.skip(), gr.skip()
            
            logger.info(f"📎 Extrayendo texto de {len(files)} archivo(s)...")
            docs_text = extract_text_from_files(files, max_chars=100000)
            
            if docs_text:
                logger.info(f"✅ Extraídos {len(docs_text)} caracteres")
                current_hist[-1][1] = "✅ Archivos procesados. Analizando con NIIF..."
                yield current_hist, "", chats, gr.skip(), gr.skip()
            else:
                logger.warning("⚠️ No se pudo extraer texto")
                docs_text = "\n[No se pudo extraer texto de los archivos]\n"
                
        except Exception as e:
            logger.exception(f"❌ Error extrayendo texto: {e}")
            docs_text = f"\n[Error procesando archivos: {str(e)[:200]}]\n"

    # Audio (placeholder)
    if audio:
        docs_text += "\n\n[Audio recibido - transcripción no implementada]"

    answer = None
    used_rag = False

    # OPCIÓN 1: RAG (si está disponible)
    if RAG_STATUS['initialized'] and conversational_rag_chain:
        try:
            logger.info("🤖 Usando RAG chain con contexto NIIF...")
            
            # Construir input mejorado con documentos
            if docs_text:
                if not user_message:
                    enhanced_input = (
                        "He subido documentos para análisis. Por favor:\n\n"
                        "1. Resume el contenido identificado\n"
                        "2. Identifica normas NIIF aplicables\n"
                        "3. Propón asientos contables (si aplica)\n"
                        "4. Indica observaciones y recomendaciones\n\n"
                        f"**DOCUMENTOS:**\n{docs_text}"
                    )
                else:
                    enhanced_input = (
                        f"{user_message}\n\n"
                        f"**--- DOCUMENTOS ADJUNTOS ---**\n"
                        f"{docs_text}\n"
                        f"**--- FIN DOCUMENTOS ---**\n\n"
                        f"Responde considerando:\n"
                        f"- Normas NIIF del contexto normativo\n"
                        f"- Información de los documentos adjuntos"
                    )
            else:
                enhanced_input = user_message
            
            # Invocar RAG
            response = conversational_rag_chain.invoke(
                {"input": enhanced_input},
                config={"configurable": {"session_id": session_id}}
            )
            
            # Extraer respuesta
            if isinstance(response, dict) and "answer" in response:
                answer = response["answer"]
            elif hasattr(response, "content"):
                answer = response.content
            else:
                answer = str(response)
            
            used_rag = True
            logger.info(f"✅ RAG respondió: {len(answer)} caracteres")
            
        except Exception as e:
            logger.exception(f"❌ Error ejecutando RAG: {e}")
            answer = None
            used_rag = False

    # OPCIÓN 2: LLM Directo (fallback)
    if answer is None and llm:
        try:
            logger.info("🔄 Usando LLM directo (sin contexto RAG completo)...")
            
            # Construir prompt con historial
            history_context = ""
            if current_hist and len(current_hist) > 2:
                recent = current_hist[-6:-2]
                for h in recent:
                    if h[0]:
                        history_context += f"\n**Usuario**: {h[0][:300]}"
                    if h[1]:
                        history_context += f"\n**Asistente**: {h[1][:300]}"

            base_prompt = (
                "Eres Jarvis, un CPA experto en NIIF para Venezuela.\n"
                "Aunque no tengo acceso completo a la base normativa en este momento, "
                "responderé con mi mejor conocimiento de NIIF.\n\n"
            )

            if docs_text:
                if not user_message:
                    full_prompt = (
                        f"{base_prompt}"
                        f"El usuario subió documentos. Analízalos y proporciona:\n"
                        f"1. Resumen del contenido\n"
                        f"2. Normas NIIF probablemente aplicables\n"
                        f"3. Tratamiento contable sugerido\n\n"
                        f"**DOCUMENTOS:**\n{docs_text[:15000]}"
                    )
                else:
                    full_prompt = (
                        f"{base_prompt}"
                        f"{history_context}\n\n"
                        f"**Usuario**: {user_message}\n\n"
                        f"**DOCUMENTOS:**\n{docs_text[:15000]}\n\n"
                        f"**Asistente**:"
                    )
            else:
                full_prompt = (
                    f"{base_prompt}"
                    f"{history_context}\n\n"
                    f"**Usuario**: {user_message}\n\n"
                    f"**Asistente**:"
                )

            answer = call_llm_for_task(llm, full_prompt, task_type="explain")
            
            if not answer or len(answer.strip()) < 10:
                answer = (
                    "⚠️ **No se pudo generar una respuesta adecuada**\n\n"
                    "Posibles causas:\n"
                    "- Filtros de seguridad del modelo\n"
                    "- Contenido no procesable\n"
                    "- Límites de contexto\n\n"
                    "**Intenta**:\n"
                    "1. Reformular la pregunta\n"
                    "2. Archivos más pequeños\n"
                    "3. Dividir en consultas simples"
                )
            else:
                # Disclaimer fallback
                answer = (
                    f"⚠️ *Respondiendo sin acceso completo a base normativa RAG*\n\n"
                    f"{answer}\n\n"
                    f"---\n"
                    f"💡 Para respuestas con citas normativas exactas, "
                    f"espera a que el sistema RAG termine de inicializar."
                )
            
            logger.info(f"✅ LLM fallback: {len(answer)} caracteres")
            
        except Exception as e:
            logger.exception(f"❌ Error en LLM fallback: {e}")
            answer = (
                f"❌ **Error procesando consulta**: {str(e)[:200]}\n\n"
                f"**Estado del sistema**: {RAG_STATUS['message']}\n\n"
                f"Por favor:\n"
                f"1. Verifica archivos válidos\n"
                f"2. Reformula la pregunta\n"
                f"3. Revisa logs del sistema"
            )

    # Si no hay nada
    if answer is None:
        answer = (
            "❌ **Sistema no disponible**\n\n"
            f"**Estado**: {RAG_STATUS['message']}\n"
            f"**Error**: {RAG_STATUS.get('error', 'Desconocido')}\n\n"
            "Verifica:\n"
            "- GOOGLE_API_KEY configurada\n"
            "- Dependencias instaladas\n"
            "- Logs para detalles"
        )

    # Actualizar historial
    current_hist.pop()  # Quitar "Procesando..."
    current_hist.append([None, answer])

    # Actualizar título si es nuevo
    if chats[current_chat_id]["title"].startswith("Nuevo chat") and user_message:
        new_title = user_message[:50] + ("..." if len(user_message) > 50 else "")
        chats[current_chat_id]["title"] = new_title

    # Actualizar UI
    samples = [[chats[id_]["title"]] for id_ in filtered_order]
    yield current_hist, "", chats, gr.Dataset(samples=samples), filtered_order

    # Auditoría
    try:
        audit_record({
            "event": "chat",
            "user_message": user_message[:200],
            "has_files": bool(files),
            "used_rag": used_rag,
            "rag_status": RAG_STATUS['message'],
            "embedding_type": RAG_STATUS.get('embedding_type', 'unknown'),
            "model": GEMINI_MODEL
        })
    except Exception:
        pass

# ------------------------------------------------------------------
# Funciones auxiliares UI
# ------------------------------------------------------------------
def _export_chat(current_chat_id, chats):
    try:
        if current_chat_id not in chats:
            return "❌ No hay chat actual para exportar"
        
        h = chats[current_chat_id]["history"] or []
        data = {
            "title": chats[current_chat_id]["title"],
            "history": h,
            "generated_at": datetime.utcnow().isoformat() + 'Z'
        }
        
        fname = f"chat_export_{uuid.uuid4().hex[:8]}.json"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))
        
        return f"✅ Chat exportado a: {fname}"
    except Exception as e:
        logger.exception("Error exportando chat")
        return f"❌ Error: {str(e)}"

def _show_extracted(files):
    if not files:
        return "ℹ️ No hay archivos para previsualizar"
    try:
        text = extract_text_from_files(files, max_chars=50000)
        if not text:
            return "⚠️ No se pudo extraer texto"
        
        preview = text[:20000]
        if len(text) > 20000:
            preview += f"\n\n... (truncado - total: {len(text)} caracteres)"
        
        return preview
    except Exception as e:
        logger.exception("Error extrayendo texto")
        return f"❌ Error: {str(e)}"

def _status():
    status = get_rag_status()
    
    # Espacio en disco
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        used_gb = used // (2**30)
        total_gb = total // (2**30)
        disk_info = f"💾 Disco: {used_gb}GB / {total_gb}GB ({free_gb}GB libres)"
    except Exception:
        disk_info = "💾 Disco: Info no disponible"
    
    # Tamaño vector store
    vector_store_size = "desconocido"
    if os.path.exists(chroma_dir):
        try:
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, dirnames, filenames in os.walk(chroma_dir)
                for filename in filenames
            )
            vector_store_size = f"{total_size / (1024**2):.1f} MB"
        except Exception:
            pass
    
    status_text = "="*60 + "\n"
    status_text += "📊 ESTADO DEL SISTEMA JARVIS\n"
    status_text += "="*60 + "\n\n"
    
    status_text += "🤖 **MODELO LLM**\n"
    status_text += f"   Modelo: {GEMINI_MODEL}\n"
    status_text += f"   Estado: {'✅ Activo' if llm else '❌ No disponible'}\n"
    status_text += f"   Temperatura: 0.3 (precisión contable)\n"
    status_text += f"   Max tokens: {DEFAULT_MAX_OUTPUT_TOKENS}\n\n"
    
    status_text += "🧠 **SISTEMA RAG**\n"
    status_text += f"   Inicializado: {'✅ Sí' if status['initialized'] else '❌ No'}\n"
    status_text += f"   Estado: {status['message']}\n"
    status_text += f"   Retriever: {'✅ Activo' if retriever else '❌ No disponible'}\n\n"
    
    status_text += "📦 **EMBEDDINGS**\n"
    status_text += f"   Tipo: {status.get('embedding_type', 'unknown').upper()}\n"
    status_text += f"   Estado: {'✅ Creados' if status.get('embeddings_created') else '⏳ Pendientes'}\n"
    if status.get('embedding_type') == 'local':
        status_text += f"   Modelo: {LOCAL_EMBEDDING_MODEL.split('/')[-1]}\n"
        status_text += f"   Device: {LOCAL_EMBEDDING_DEVICE}\n"
        status_text += f"   💰 Gratuito e ilimitado\n"
    if status.get('progress'):
        status_text += f"   Progreso: {status['progress']}\n"
    status_text += "\n"
    
    status_text += "📚 **BASE DE CONOCIMIENTOS**\n"
    status_text += f"   Documentos: {len(todos_los_fragmentos)} fragmentos NIIF\n"
    status_text += f"   ChromaDB: {chroma_dir}\n"
    status_text += f"   Tamaño: {vector_store_size}\n\n"
    
    status_text += "💾 **RECURSOS**\n"
    status_text += f"   {disk_info}\n\n"
    
    if status.get('error'):
        status_text += "❌ **ERROR**\n"
        status_text += f"   {status['error']}\n\n"
    
    status_text += "="*60 + "\n"
    
    return status_text

def _new_chat(chats, chat_order):
    new_id = str(uuid.uuid4())
    chats[new_id] = {
        "title": f"Nuevo chat {len(chats) + 1}",
        "history": WELCOME_MESSAGE.copy()
    }
    chat_order = chat_order + [new_id]
    samples = [[chats[id_]["title"]] for id_ in chat_order]
    return chats, new_id, WELCOME_MESSAGE.copy(), samples, chat_order, chat_order

def _switch_chat(evt: gr.SelectData, filtered_order, chats):
    if evt.index is None or evt.index >= len(filtered_order):
        return None, None
    chat_id = filtered_order[evt.index]
    return chats[chat_id]["history"], chat_id

def _filter_chats(query, chats, chat_order):
    if not query:
        filtered = chat_order
    else:
        filtered = [
            id_ for id_ in chat_order 
            if query.lower() in chats[id_]["title"].lower()
        ]
    samples = [[chats[id_]["title"]] for id_ in filtered]
    return samples, filtered

def _clear_current(current_chat_id, chats, filtered_order):
    if current_chat_id in chats:
        chats[current_chat_id]["history"] = []
    samples = [[chats[id_]["title"]] for id_ in filtered_order]
    return [], chats, gr.Dataset(samples=samples)

# ------------------------------------------------------------------
# MAIN: Lanzar aplicación
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Limpiar archivos temporales al inicio
    logger.info("🧹 Limpiando archivos temporales al inicio...")
    cleanup_temp_files()
    
    # Verificaciones iniciales
    print("\n" + "="*80)
    print("🚀 INICIANDO JARVIS - ASISTENTE CONTABLE NIIF")
    print("="*80)
    
    if not GOOGLE_APPLICATION_CREDENTIALS:
        logger.warning("ℹ️ GOOGLE_APPLICATION_CREDENTIALS no configurado")
        print("⚠️  Google Vision OCR no disponible (solo afecta OCR de imágenes)")
    
    if not GOOGLE_API_KEY:
        logger.error("❌ GOOGLE_API_KEY no configurada")
        print("\n" + "="*80)
        print("⚠️  ADVERTENCIA CRÍTICA")
        print("="*80)
        print("GOOGLE_API_KEY no está configurada.")
        print("El sistema no funcionará correctamente sin esta variable.")
        print("")
        print("Configúrala en tu archivo .env o como variable de entorno:")
        print("GOOGLE_API_KEY=tu_api_key_aqui")
        print("="*80 + "\n")
    
    if not LANGCHAIN_IMPORTS_OK:
        logger.error("❌ LangChain/Google-GenAI imports faltantes")
        print("\n" + "="*80)
        print("⚠️  DEPENDENCIAS FALTANTES")
        print("="*80)
        print("Instala las dependencias necesarias:")
        print("pip install langchain langchain-google-genai langchain-community chromadb")
        print("="*80 + "\n")
    
    if not LOCAL_EMBEDDINGS_AVAILABLE:
        logger.warning("⚠️ Embeddings locales no disponibles")
        print("\n" + "="*80)
        print("⚠️  EMBEDDINGS LOCALES RECOMENDADOS")
        print("="*80)
        print("Para evitar límites de cuota de Google, instala:")
        print("pip install sentence-transformers torch")
        print("")
        print("Esto permite:")
        print("- Procesamiento ilimitado y gratuito")
        print("- Sin cuotas de API para vectorización")
        print("- Funcionamiento offline")
        print("="*80 + "\n")
    
    # Verificar espacio en disco
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (2**30)
        if free_gb < 2:
            logger.warning(f"⚠️ ESPACIO BAJO: Solo {free_gb}GB libres")
            print("\n" + "="*80)
            print("⚠️  ADVERTENCIA: POCO ESPACIO EN DISCO")
            print("="*80)
            print(f"Espacio libre: {free_gb}GB")
            print("")
            print("Recomendaciones:")
            print("1. Usa modelo ligero: USE_LIGHTWEIGHT_MODEL=true")
            print("2. Aumenta CHUNK_SIZE para menos fragmentos")
            print("3. Limpia archivos temporales con el botón en la UI")
            print("4. Considera eliminar vector store anterior si existe")
            print("="*80 + "\n")
        else:
            print(f"✅ Espacio en disco: {free_gb}GB disponibles")
    except Exception:
        pass
    
    # Resumen de configuración
    print("\n" + "="*80)
    print("📋 CONFIGURACIÓN ACTUAL")
    print("="*80)
    print(f"🤖 Modelo LLM: {GEMINI_MODEL}")
    print(f"🧠 Embeddings: {'Local (' + LOCAL_EMBEDDING_MODEL.split('/')[-1] + ')' if LOCAL_EMBEDDINGS_AVAILABLE else 'No disponibles'}")
    print(f"📦 Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP}")
    print(f"📂 ChromaDB: {chroma_dir}")
    print(f"📚 PDFs configurados: {len(rutas_documentos)}")
    print(f"🪶 Modo ligero: {'✅ Activado' if USE_LIGHTWEIGHT_MODEL else '❌ Desactivado'}")
    print("="*80 + "\n")
    
    if gr is None:
        print("❌ Gradio no está instalado. Ejecuta: pip install gradio")
        sys.exit(1)

    # CSS mejorado
    custom_css = '''
    body { 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .gradio-container { 
        max-width: 1400px !important;
        margin: 20px auto !important;
        background: #ffffff !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3) !important;
        overflow: hidden !important;
    }
    
    .header {
        padding: 24px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 16px 16px 0 0;
        text-align: center;
    }
    
    .header h1 {
        margin: 0;
        font-size: 32px;
        font-weight: 700;
    }
    
    .header p {
        margin: 8px 0 0 0;
        opacity: 0.9;
    }
    
    .sidebar {
        background: #f8fafc;
        border-right: 2px solid #e2e8f0;
        padding: 20px;
        border-radius: 0 0 0 16px;
        height: 100%;
    }
    
    .chat-column {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 240px);
        max-height: calc(100vh - 240px);
        padding: 20px;
        overflow: hidden;
    }
    
    .chatbot {
        flex: 1;
        overflow-y: auto;
        background: #ffffff;
        margin-bottom: 16px;
        min-height: 0;
    }
    
    .chatbot .message {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    .chatbot .message-row {
        margin-bottom: 12px !important;
    }
    
    .input-area-container {
        flex-shrink: 0;
        background: #f8fafc;
        border-top: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
        margin-top: 0;
    }
    
    .attach-buttons {
        display: flex;
        gap: 8px;
        margin-bottom: 8px;
    }
    
    .attach-btn {
        padding: 6px 12px !important;
        font-size: 13px !important;
        border-radius: 8px !important;
        background: #e2e8f0 !important;
        transition: all 0.2s !important;
    }
    
    .attach-btn:hover {
        background: #cbd5e1 !important;
        transform: translateY(-1px);
    }
    
    .input-area-container textarea {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 12px !important;
        font-size: 15px !important;
        resize: none !important;
        width: 100% !important;
    }
    
    .input-area-container textarea:focus {
        border-color: #667eea !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    .send-button {
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        max-width: 40px !important;
        padding: 0 !important;
        margin-left: 8px !important;
        border-radius: 10px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
        flex-shrink: 0 !important;
    }
    
    .send-button:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4) !important;
    }
    
    .send-button::before {
        content: "✈️";
        font-size: 16px;
        display: block;
    }
    
    .send-button span {
        display: none !important;
    }
    
    .input-row {
        display: flex !important;
        align-items: flex-end !important;
        gap: 0 !important;
        width: 100% !important;
    }
    
    .input-text {
        flex: 1 !important;
        min-width: 0 !important;
    }
    
    .upload-btn, .system-btn {
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    .upload-btn:hover, .system-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .footer {
        text-align: center;
        padding: 12px;
        color: #64748b;
        font-size: 13px;
        background: #f8fafc;
        border-radius: 0 0 16px 16px;
        margin-top: 0;
        flex-shrink: 0;
    }
    
    .gradio-container .contain {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow: hidden !important;
    }
    
    .gradio-container .main {
        height: 100% !important;
        overflow-y: auto !important;
    }
    
    .gradio-container > div {
        min-height: auto !important;
    }
    '''

    # Construir interfaz
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Jarvis - Asistente Contable NIIF") as demo:
        # Estados
        chats = gr.State({"chat1": {"title": "Chat principal", "history": WELCOME_MESSAGE.copy()}})
        current_chat_id = gr.State("chat1")
        chat_order = gr.State(["chat1"])
        filtered_order = gr.State(["chat1"])

        # Header
        gr.HTML(f"""
            <div class='header'>
                <h1>🤖 Jarvis — Asistente Contable NIIF</h1>
                <p>Tu experto en Normas Internacionales de Información Financiera para Venezuela</p>
                <small style='opacity: 0.8;'>🧠 Powered by {GEMINI_MODEL} + Local Embeddings</small>
            </div>
        """)

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1, elem_classes="sidebar"):
                gr.Markdown("### 💬 Conversaciones")
                new_chat_btn = gr.Button("➕ Nuevo Chat", variant="primary")
                search_box = gr.Textbox(
                    placeholder="🔍 Buscar conversaciones...",
                    show_label=False,
                    container=False
                )
                chat_dataset = gr.Dataset(
                    headers=["Título"],
                    samples=[["Chat principal"]],
                    type="index"
                )
                
                gr.Markdown("---")
                gr.Markdown("### ⚙️ Sistema")
                status_btn = gr.Button("📊 Ver Estado", size="sm")
                cleanup_btn = gr.Button("🧹 Limpiar Espacio", size="sm", variant="secondary")
                
                status_output = gr.Textbox(
                    label="Estado del Sistema",
                    lines=20,
                    max_lines=30,
                    visible=False
                )
                
                cleanup_output = gr.Textbox(
                    label="Resultado de Limpieza",
                    lines=3,
                    visible=False
                )

            # Chat area
            with gr.Column(scale=4, elem_classes="chat-column"):
                chat = gr.Chatbot(
                    value=WELCOME_MESSAGE.copy(),
                    show_label=False,
                    height=500,
                    avatar_images=(
                        None,
                        "https://api.dicebear.com/7.x/bottts/svg?seed=Jarvis"
                    ),
                    show_copy_button=True,
                    layout="panel",
                    bubble_full_width=False
                )

                # Área de input
                with gr.Group(elem_classes="input-area-container"):
                    # Botones para adjuntar
                    with gr.Row(elem_classes="attach-buttons"):
                        attach_file_btn = gr.Button("📎 Adjuntar Archivo", size="sm", elem_classes="attach-btn")
                        attach_audio_btn = gr.Button("🎤 Grabar Audio", size="sm", elem_classes="attach-btn")
                    
                    # Inputs colapsables
                    file_upload = gr.File(
                        file_types=[".pdf", ".docx", ".png", ".jpg", ".jpeg", ".txt"],
                        file_count="multiple",
                        type="filepath",
                        label="Archivos",
                        visible=False
                    )
                    
                    audio_upload = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="Audio",
                        visible=False
                    )

                    # Input principal con botón de envío
                    with gr.Row(elem_classes="input-row"):
                        user_input = gr.Textbox(
                            placeholder="Pregunta sobre NIIF, sube facturas para análisis, pide asientos contables...",
                            show_label=False,
                            lines=2,
                            max_lines=4,
                            autofocus=True,
                            elem_classes="input-text",
                            container=False
                        )
                        
                        send = gr.Button("", elem_classes="send-button", size="sm")

        # Footer
        gr.HTML(f"""
            <div class='footer'>
                <strong>Jarvis v2.1 - Gemini 2.5 Pro Edition</strong> | Desarrollado con ❤️ usando Gradio, LangChain y Google Gemini
                <br>
                <small>🧠 Embeddings locales ilimitados | 💰 Sin cuotas de API | 🤖 Modelo: {GEMINI_MODEL}</small>
            </div>
        """)

        # Event handlers
        
        # Toggle para mostrar/ocultar inputs de archivos
        def toggle_file_upload(visible_state):
            return gr.update(visible=not visible_state), not visible_state
        
        def toggle_audio_upload(visible_state):
            return gr.update(visible=not visible_state), not visible_state
        
        file_visible = gr.State(False)
        audio_visible = gr.State(False)
        
        attach_file_btn.click(
            toggle_file_upload,
            inputs=[file_visible],
            outputs=[file_upload, file_visible]
        )
        
        attach_audio_btn.click(
            toggle_audio_upload,
            inputs=[audio_visible],
            outputs=[audio_upload, audio_visible]
        )
        
        # Enviar con clic
        send.click(
            _chat_fn,
            inputs=[user_input, chat, file_upload, audio_upload, current_chat_id, chats, filtered_order],
            outputs=[chat, user_input, chats, chat_dataset, filtered_order]
        )

        # Enviar con Enter
        user_input.submit(
            _chat_fn,
            inputs=[user_input, chat, file_upload, audio_upload, current_chat_id, chats, filtered_order],
            outputs=[chat, user_input, chats, chat_dataset, filtered_order]
        )

        new_chat_btn.click(
            _new_chat,
            inputs=[chats, chat_order],
            outputs=[chats, current_chat_id, chat, chat_dataset, chat_order, filtered_order]
        )

        chat_dataset.select(
            _switch_chat,
            inputs=[filtered_order, chats],
            outputs=[chat, current_chat_id]
        )

        search_box.change(
            _filter_chats,
            inputs=[search_box, chats, chat_order],
            outputs=[chat_dataset, filtered_order]
        )

        status_btn.click(
            lambda: (gr.update(visible=True), _status()),
            inputs=None,
            outputs=[status_output, status_output]
        )
        
        cleanup_btn.click(
            lambda: (gr.update(visible=True), f"🧹 Limpiados {cleanup_temp_files()} archivos temporales"),
            inputs=None,
            outputs=[cleanup_output, cleanup_output]
        )

    # Lanzar aplicación
    logger.info("🚀 Lanzando interfaz Gradio...")
    print("\n" + "="*80)
    print("✅ INTERFAZ LISTA - Iniciando servidor...")
    print("="*80 + "\n")
    
    demo.queue(default_concurrency_limit=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True
    )
 
