"""
TechParts Chat – Backend Flask + Amazon Bedrock Agents Runtime
Suporta texto, imagens e documentos (upload via S3).
Run:  python app.py
"""

import base64
import json
import logging
import mimetypes
import os
import uuid
from datetime import datetime, timezone

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from flask import Flask, jsonify, request
from flask_cors import CORS

# ──────────────────────────────────────────
# Config
# ──────────────────────────────────────────
AGENT_ID       = "MGUOGTRRAL"
AGENT_ALIAS_ID = "CFBBT4ALDX"
REGION         = "us-east-1"
PORT           = 5000

# Modelo com suporte a visão, usado para analisar imagens via Converse API
# (o agente Bedrock Classic com Code Interpreter não suporta tipos de imagem
# em sessionState.files — apenas CSV/XLS/XLSX/YAML/JSON/DOC/DOCX/HTML/MD/TXT/PDF)
VISION_MODEL_ID = "arn:aws:bedrock:us-east-1:608040300344:inference-profile/us.anthropic.claude-sonnet-4-6"

# Knowledge Base
KNOWLEDGE_BASE_ID = "PT9DQ9KBJK"

# S3 (mantido para possível uso futuro / arquivos grandes)
S3_BUCKET = "personal-ai-agent-media"
S3_PREFIX = "chat-attachments"

# Upload limits
MAX_FILE_SIZE_MB = 10

# Tipos de imagem: NÃO suportados por sessionState.files do agente.
# Serão analisados via Bedrock Runtime Converse API (visão nativa do modelo).
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}

# Tipos de documento: suportados nativamente por sessionState.files (Code Interpreter).
ALLOWED_DOC_TYPES = {
    "application/pdf",
    "text/plain",
    "text/csv",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

# Mapeamento MIME -> formato aceito pela Converse API
_IMAGE_FORMAT_MAP = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/gif": "gif",
    "image/webp": "webp",
}

# Mapeamento MIME -> extensão aceita por sessionState.files (Code Interpreter)
_DOC_EXTENSION_MAP = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/csv": "csv",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

# ──────────────────────────────────────────
# Logging
# ──────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ──────────────────────────────────────────
# Flask app
# ──────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max payload

# Allow requests from the local frontend (any origin on localhost)
CORS(app, resources={r"/*": {"origins": ["http://localhost:*", "http://127.0.0.1:*", "null"]}})

# ──────────────────────────────────────────
# Bedrock clients (lazy singletons)
# ──────────────────────────────────────────
_bedrock_agent_client = None
_bedrock_runtime_client = None
_s3_client = None

def get_bedrock_client():
    """Return a cached bedrock-agent-runtime client."""
    global _bedrock_agent_client
    if _bedrock_agent_client is None:
        _bedrock_agent_client = boto3.client(
            "bedrock-agent-runtime",
            region_name=REGION,
        )
    return _bedrock_agent_client


def get_bedrock_kb_client():
    """Return a cached bedrock-agent-runtime client (para retrieve da KB)."""
    return get_bedrock_client()


def get_bedrock_runtime_client():
    """Return a cached bedrock-runtime client (para Converse API com visão)."""
    global _bedrock_runtime_client
    if _bedrock_runtime_client is None:
        _bedrock_runtime_client = boto3.client(
            "bedrock-runtime",
            region_name=REGION,
        )
    return _bedrock_runtime_client


def get_s3_client():
    """Return a cached S3 client."""
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=REGION)
    return _s3_client


# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def retrieve_from_knowledge_base(query: str) -> str:
    """
    Consulta a Knowledge Base do Bedrock e retorna os trechos relevantes.
    """
    client = get_bedrock_kb_client()

    try:
        response = client.retrieve(
            knowledgeBaseId=KNOWLEDGE_BASE_ID,
            retrievalQuery={"text": query},
            retrievalConfiguration={
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                }
            },
        )

        results = response.get("retrievalResults", [])
        if not results:
            return ""

        context_parts = []
        for i, result in enumerate(results, 1):
            content = result.get("content", {}).get("text", "")
            source = result.get("location", {}).get("s3Location", {}).get("uri", "N/A")
            if content:
                context_parts.append(f"[Fonte {i}]: {content}")

        return "\n\n".join(context_parts)

    except (ClientError, BotoCoreError) as exc:
        log.warning("Erro ao consultar Knowledge Base: %s", exc)
        return ""


def invoke_agent(message: str, session_id: str) -> str:
    """
    Call Bedrock Agents Runtime invoke_agent and collect the full response.
    """
    client = get_bedrock_client()

    log.info("Invoking agent | session=%s | message=%r", session_id, message[:120])

    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=session_id,
        inputText=message,
        enableTrace=False,
    )

    # The response body is a streaming EventStream
    completion = ""
    event_stream = response.get("completion", [])
    for event in event_stream:
        chunk = event.get("chunk")
        if chunk:
            raw = chunk.get("bytes", b"")
            completion += raw.decode("utf-8", errors="replace")

    log.info("Agent replied (%d chars)", len(completion))
    return completion.strip() or "(O agente não retornou uma resposta.)"


def upload_file_to_s3(file_data: bytes, filename: str, mime_type: str, session_id: str) -> str:
    """
    Faz upload de um arquivo para S3 e retorna a URI s3://.
    """
    s3 = get_s3_client()

    now = datetime.now(timezone.utc)
    key = (
        f"{S3_PREFIX}/{now.strftime('%Y/%m/%d')}/{session_id}/"
        f"{uuid.uuid4().hex[:8]}_{filename}"
    )

    s3.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=file_data,
        ContentType=mime_type,
    )

    s3_uri = f"s3://{S3_BUCKET}/{key}"
    log.info("Uploaded file to %s", s3_uri)
    return s3_uri


def analyze_image_with_vision(image_bytes: bytes, mime_type: str, filename: str, user_message: str) -> str:
    """
    Analisa uma imagem usando o Bedrock Runtime Converse API (visão nativa do Claude).
    O Bedrock Agent (Classic) não suporta imagens em sessionState.files — apenas
    o Converse API do bedrock-runtime tem suporte multimodal de visão.
    """
    client = get_bedrock_runtime_client()
    image_format = _IMAGE_FORMAT_MAP.get(mime_type, "jpeg")

    prompt_text = (
        user_message
        if user_message
        else "Analise esta imagem detalhadamente. Descreva o produto, "
             "possíveis defeitos visíveis, número de série (se legível) e "
             "condição geral, conforme os critérios de análise de imagens para RMA."
    )

    log.info("Analisando imagem '%s' via Converse API (visão)", filename)

    response = client.converse(
        modelId=VISION_MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "image": {
                            "format": image_format,
                            "source": {"bytes": image_bytes},
                        }
                    },
                    {"text": prompt_text},
                ],
            }
        ],
        inferenceConfig={"maxTokens": 1024, "temperature": 0.3},
    )

    output_message = response.get("output", {}).get("message", {})
    content_blocks = output_message.get("content", [])
    text_parts = [block["text"] for block in content_blocks if "text" in block]
    analysis = "\n".join(text_parts).strip()

    log.info("Análise de imagem concluída (%d chars)", len(analysis))
    return analysis or "(Não foi possível analisar a imagem.)"


def invoke_agent_with_files(message: str, session_id: str, files: list) -> str:
    """
    Invoca o agente com arquivos anexados.

    Estratégia:
    - Imagens (PNG/JPEG/GIF/WebP): NÃO são suportadas por sessionState.files do
      Agent (Code Interpreter só aceita CSV/XLS/XLSX/YAML/JSON/DOC/DOCX/HTML/MD/TXT/PDF).
      São analisadas separadamente via Converse API (visão) e o resultado textual
      é incluído na mensagem enviada ao agente.
    - Documentos (PDF/TXT/CSV/DOC/DOCX): enviados via sessionState.files,
      que é o mecanismo nativo e suportado pelo Code Interpreter do agente.
    """
    client = get_bedrock_client()

    log.info(
        "Invoking agent with %d file(s) | session=%s | message=%r",
        len(files), session_id, message[:120],
    )

    image_analyses = []
    doc_files = []  # arquivos elegíveis para sessionState.files

    for file_info in files:
        filename = file_info["name"]
        mime_type = file_info["type"]
        data_bytes = base64.b64decode(file_info["data"])

        if mime_type in ALLOWED_IMAGE_TYPES:
            try:
                analysis = analyze_image_with_vision(data_bytes, mime_type, filename, message)
                image_analyses.append(
                    f"[Análise da imagem '{filename}']\n{analysis}"
                )
            except (ClientError, BotoCoreError) as exc:
                log.error("Falha ao analisar imagem '%s': %s", filename, exc)
                image_analyses.append(
                    f"[Não foi possível analisar a imagem '{filename}': {exc}]"
                )
        elif mime_type in ALLOWED_DOC_TYPES:
            doc_files.append({
                "name": filename,
                "source": {
                    "sourceType": "BYTE_CONTENT",
                    "byteContent": {
                        "data": data_bytes,
                        "mediaType": mime_type,
                    },
                },
                "useCase": "CODE_INTERPRETER",
            })

    # Montar a mensagem final para o agente
    parts = []
    if message:
        parts.append(message)
    if image_analyses:
        parts.append("\n\n".join(image_analyses))
    if not parts:
        parts.append("Analise o(s) arquivo(s) enviado(s) conforme suas instruções.")

    input_text = "\n\n".join(parts)

    invoke_kwargs = {
        "agentId": AGENT_ID,
        "agentAliasId": AGENT_ALIAS_ID,
        "sessionId": session_id,
        "inputText": input_text,
        "enableTrace": False,
    }
    if doc_files:
        invoke_kwargs["sessionState"] = {"files": doc_files}

    try:
        response = client.invoke_agent(**invoke_kwargs)
    except Exception as exc:
        log.warning(
            "sessionState.files falhou (%s) — reenviando apenas texto",
            type(exc).__name__,
        )
        response = client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=input_text,
            enableTrace=False,
        )

    # Coletar resposta
    completion = ""
    event_stream = response.get("completion", [])
    for event in event_stream:
        chunk = event.get("chunk")
        if chunk:
            raw = chunk.get("bytes", b"")
            completion += raw.decode("utf-8", errors="replace")

    log.info("Agent replied (%d chars)", len(completion))
    return completion.strip() or "(O agente não retornou uma resposta.)"


# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    """Simple liveness probe."""
    return jsonify({"status": "ok"}), 200


@app.route("/chat", methods=["POST"])
def chat():
    """
    Receive a chat message (with optional file attachments) and return the agent's reply.

    Request body (JSON):
        {
            "message":    "Olá, quais peças estão disponíveis?",
            "session_id": "optional-uuid-for-multi-turn",
            "files": [
                {
                    "name": "foto_defeito.jpg",
                    "type": "image/jpeg",
                    "data": "<base64-encoded>"
                }
            ]
        }

    Response (JSON):
        {
            "response":   "...",
            "session_id": "uuid-used"
        }
    """
    body = request.get_json(silent=True)

    if not body:
        return jsonify({"error": "Body JSON é obrigatório."}), 400

    message    = str(body.get("message", "")).strip()
    session_id = str(body.get("session_id") or uuid.uuid4())
    files      = body.get("files", [])

    if not message and not files:
        return jsonify({"error": "Envie uma mensagem ou arquivo."}), 400

    # Validar arquivos
    for f in files:
        if f.get("type") not in ALLOWED_IMAGE_TYPES and f.get("type") not in ALLOWED_DOC_TYPES:
            return jsonify({
                "error": f"Tipo de arquivo não suportado: {f.get('type')}. "
                         f"Envie imagens (PNG, JPG, GIF, WebP) ou documentos (PDF, TXT, CSV, DOC, DOCX)."
            }), 400
        # Verificar tamanho (base64 é ~33% maior que o binário)
        data_size = len(f.get("data", "")) * 3 / 4
        if data_size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return jsonify({"error": f"Arquivo {f.get('name')} excede o limite de {MAX_FILE_SIZE_MB}MB."}), 400

    try:
        if files:
            reply = invoke_agent_with_files(message, session_id, files)
        else:
            reply = invoke_agent(message, session_id)

        return jsonify({"response": reply, "session_id": session_id}), 200

    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        error_msg  = exc.response["Error"]["Message"]
        log.error("Bedrock ClientError [%s]: %s", error_code, error_msg)

        friendly = {
            "AccessDeniedException":   "Sem permissão para invocar o agente. Verifique as credenciais AWS e a política IAM.",
            "ResourceNotFoundException": "Agente não encontrado. Confirme o Agent ID e Alias ID.",
            "ThrottlingException":     "Muitas requisições. Aguarde um momento e tente novamente.",
            "ValidationException":     f"Parâmetro inválido: {error_msg}",
        }.get(error_code, f"Erro AWS [{error_code}]: {error_msg}")

        return jsonify({"error": friendly}), 502

    except BotoCoreError as exc:
        log.error("BotoCoreError: %s", exc)
        return jsonify({"error": f"Erro interno ao chamar o Bedrock: {exc}"}), 500

    except Exception as exc:  # noqa: BLE001
        log.exception("Unexpected error")
        return jsonify({"error": f"Erro inesperado: {exc}"}), 500


# ──────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting TechParts Chat backend on http://localhost:%d", PORT)
    log.info("Agent ID: %s  |  Alias: %s  |  Region: %s", AGENT_ID, AGENT_ALIAS_ID, REGION)
    app.run(host="0.0.0.0", port=PORT, debug=False)
