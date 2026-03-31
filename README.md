## Team Assistant Telegram Bot (Haystack + Pinecone + OpenAI)

### Что делает

- **Слушает чат** между командами `/start_listening` и `/stop_listening`
- **Индексирует каждое сообщение** в Pinecone (с метаданными автора/времени/чата)
- **Отвечает на упоминание (@mention)**, используя:
  - последние **100** сообщений из чата как локальный контекст
  - поиск по Pinecone через Haystack (RAG)
- **На /stop_listening**: делает итоговое резюме с “мнениями/решениями/next steps” и сохраняет саммари в Pinecone

### Установка

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

### Настройка

Скопируйте `.env.example` в `.env` и заполните значения:

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME` (по умолчанию `team_chat`, если индекса нет — будет создан)
- `OPENAI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `OPENAI_MODEL` (по умолчанию `gpt-4o-mini`)
- `EMBEDDING_MODEL` (по умолчанию `text-embedding-3-small`)

Дополнительно (опционально):

- `PINECONE_CLOUD` (по умолчанию `aws`)
- `PINECONE_REGION` (по умолчанию `us-east-1`)
- `TOP_K` (по умолчанию `8`)
- `LAST_MESSAGES_CONTEXT` (по умолчанию `100`)
- `LOG_LEVEL` (по умолчанию `INFO`)
- `LOG_FORMAT` (по умолчанию `%(asctime)s %(levelname)s %(message)s`)
- `LOG_DATEFMT` (по умолчанию не задан)

### Запуск

```bash
python run.py
```

Если вы запускаете НЕ из virtualenv (например, просто `python3 run.py`), то возможны ошибки вида
`UnicodeEncodeError: 'latin-1' codec can't encode ...` при записи кириллицы в Pinecone.
Запускайте строго из `.venv` и установите зависимости через `requirements.txt`.

### Тесты

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
pytest -q
```

Покрытие тестами (что проверяется):

- `tests/test_config.py`
  - валидация обязательных переменных окружения (`PINECONE_API_KEY`, `OPENAI_API_KEY`, `TELEGRAM_BOT_TOKEN`)
  - валидация имени индекса `PINECONE_INDEX_NAME` (только `a-z0-9-`)
- `tests/test_metadata_filters.py`
  - корректный формат фильтров Haystack для Pinecone (`{"field": "meta.<...>", "operator": "==", "value": ...}`)
- `tests/test_rag_build_services.py`
  - smoke-test сборки пайплайнов (`build_services`) без сетевых вызовов (OpenAI/Pinecone не дергаются)

### Директория `data/` (создается автоматически)

`data/` — локальное состояние бота на диске (нужно, чтобы помнить активные сессии и хранить полный лог переписки между `/start_listening` и `/stop_listening` без ограничений по количеству сообщений).

- `data/state.json`
  - состояние по каждому `chat_id`
  - поля: `listening` (включён ли режим сессии) и `session_id` (текущая/последняя сессия)
- `data/sessions/<chat_id>/<session_id>.jsonl`
  - полный лог сообщений конкретной сессии в формате JSONL (1 строка = 1 сообщение с метаданными автора/времени/текста)
  - используется для итогового анализа на `/stop_listening`

### Команды в Telegram

- `/start_listening` — начать сохранять и индексировать все последующие сообщения
- `/stop_listening` — остановить, сделать итоговое резюме и сохранить его

### Обработка ошибок (коротко)

- Ошибки Pinecone/OpenAI при индексинге сообщений не должны “ронять” polling: бот логирует исключение и продолжает.
- На mention и на `/stop_listening` ошибки оборачиваются в понятный ответ в чат + подробности в логах.

