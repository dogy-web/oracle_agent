# MOS Agent (Oracle Support Copilot)

Servicio FastAPI que combina Playwright + OpenAI para consultar [My Oracle Support](https://support.oracle.com/) (MOS), generar búsquedas guiadas y devolver documentos relevantes. Incluye endpoints compatibles con herramientas de LLMs (function calling) y un set de capturas de referencia.

## Componentes principales

- `mos_agent.py`: API + lógica del agente. Expone funciones `search_mos`, `search_mos_from_log` y `get_document`.
- `requirements.txt`: fastapi, uvicorn, playwright y OpenAI SDK.
- `recording.json`: ejemplo de interacción (útil para testing).
- `screenshot*.png` / `search*.png`: capturas de UI para depuración.

## Variables de entorno

| Variable | Uso |
| --- | --- |
| `MOS_LOGIN_USER`, `MOS_LOGIN_PASSWORD` | Credenciales de Oracle Support |
| `MOS_PROFILE_DIR` | Perfil Chromium persistente (para evitar MFA reiterado) |
| `MOS_PAGE_TIMEOUT_MS` | Timeout per page en Playwright |
| `MOS_HEADLESS` | `1` para headless, `0` para ver el navegador |
| `OPENAI_API_KEY` / `OPENAI_MODEL` | Motor LLM para razonar sobre los resultados |

## Ejecución local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
uvicorn mos_agent:app --reload --port 8000
```

Una vez arriba, podés probar con:

```bash
curl -X POST http://localhost:8000/search \
     -H "Content-Type: application/json" \
     -d '{"queries": ["ORA-7445 core dump"]}'
```

El agente abrirá MOS, ejecutará las búsquedas solicitadas y devolverá los títulos + enlaces listos para consumir desde tu flujo de LLM.
