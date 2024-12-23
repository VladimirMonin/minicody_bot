"""
Telegram-бот для поддержки студентов в группах с возможностью обработки текста и изображений.
Использует GPT-4 Vision API для анализа изображений и ответов на вопросы.

Основные возможности:
- Обработка текстовых сообщений с обращением к боту
- Анализ изображений через Vision API
- Поддержка контекста сообщений
- Ограничение количества сообщений в сутки
- Асинхронная обработка запросов
"""

import json
import base64
import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from openai import AsyncOpenAI
from telegram.constants import ParseMode


from settings import (
    ALLOWED_CHATS,
    MAX_MESSAGES_PER_DAY,
    CONTEXT_EXPIRATION_MINUTES,
    CONTEXT_MESSAGE_LIMIT,
    JSON_LOG_FILE,
    OPEN_AI_API_KEY,
    BOT_TOKEN,
    BOT_ROLE,
    MODEL,
)

# Настройка логгера
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Глобальные переменные
chat_logs: Dict[int, Dict[int, List[Dict[str, str]]]] = {}
message_counters: Dict[int, Dict[int, int]] = {}

# Инициализация клиента OpenAI
openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)

# Создание временной директории для изображений
TEMP_IMAGE_DIR = Path("temp_images")
TEMP_IMAGE_DIR.mkdir(exist_ok=True)


async def encode_image(image_path: str) -> str:
    """Кодирует изображение в base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

async def save_temp_image(photo_file: Any, file_name: str) -> Path:
    """Сохраняет изображение во временную директорию."""
    file_path = TEMP_IMAGE_DIR / file_name
    await photo_file.download_to_drive(file_path)
    return file_path

async def cleanup_temp_images(image_paths: List[Path]) -> None:
    """Удаляет временные файлы изображений."""
    for path in image_paths:
        try:
            path.unlink()
        except Exception as e:
            logger.error(f"Ошибка при удалении {path}: {e}")

async def load_chat_logs() -> None:
    """Загружает логи общения из JSON файла."""
    global chat_logs
    try:
        with open(JSON_LOG_FILE, "r", encoding="utf-8") as file:
            chat_logs = json.load(file)
    except FileNotFoundError:
        logger.info("JSON файл с логами не найден. Создаем новый.")
        chat_logs = {}

async def save_chat_logs() -> None:
    """Сохраняет текущие логи общения в JSON файл."""
    with open(JSON_LOG_FILE, "w", encoding="utf-8") as file:
        json.dump(chat_logs, file, indent=4, ensure_ascii=False)

async def get_user_context(chat_id: int, user_id: int) -> List[Dict[str, str]]:
    """Возвращает контекст последних сообщений пользователя."""
    if chat_id in chat_logs and user_id in chat_logs[chat_id]:
        context = []
        for msg in chat_logs[chat_id][user_id][-CONTEXT_MESSAGE_LIMIT:]:
            if time.time() - msg["timestamp"] < CONTEXT_EXPIRATION_MINUTES * 60:
                context.append(msg)
        return context
    return []

async def update_user_context(chat_id: int, user_id: int, message: str, role: str = "user") -> None:
    """Обновляет контекст пользователя новым сообщением."""
    timestamp = time.time()
    if chat_id not in chat_logs:
        chat_logs[chat_id] = {}
    if user_id not in chat_logs[chat_id]:
        chat_logs[chat_id][user_id] = []
    chat_logs[chat_id][user_id].append({
        "timestamp": timestamp,
        "message": message,
        "human_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "role": role
    })
    await save_chat_logs()

async def is_allowed_chat(chat_id: int) -> bool:
    """Проверяет, разрешен ли чат для работы бота."""
    return chat_id in ALLOWED_CHATS

async def is_admin(user_status: str) -> bool:
    """Проверяет, является ли пользователь администратором."""
    return user_status in ["creator", "administrator"]

async def check_message_limit(chat_id: int, user_id: int, is_admin: bool) -> bool:
    """Проверяет, не превышен ли лимит сообщений для пользователя."""
    if is_admin:
        return True
    return message_counters.get(chat_id, {}).get(user_id, 0) <= MAX_MESSAGES_PER_DAY


async def prepare_content_for_api(
    message_text: str, photos: Optional[List[Any]] = None
) -> List[Dict[str, Any]]:
    content = [{"type": "text", "text": message_text}]
    image_paths = []

    if photos:
        for i, photo in enumerate(photos):
            try:
                file = await photo.get_file()
                file_path = await save_temp_image(file, f"temp_image_{i}.jpg")
                image_paths.append(file_path)
                logger.info(f"Сохранено изображение {i+1}: {file_path}")

                base64_image = await encode_image(str(file_path))
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "auto",
                        },
                    }
                )
                logger.info(f"Изображение {i+1} закодировано в base64")
            except Exception as e:
                logger.error(f"Ошибка при обработке изображения {i+1}: {e}")

    return content, image_paths

async def get_ai_response(context_messages: List[Dict[str, str]], content: List[Dict[str, Any]]) -> str:
    """Получает ответ от OpenAI API."""
    messages = [{"role": "system", "content": BOT_ROLE}]
    
    # Добавляем контекст предыдущих сообщений
    for msg in context_messages:
        messages.append({
            "role": msg.get("role", "user"),
            "content": msg["message"]
        })
    
    # Добавляем текущее сообщение с контентом
    messages.append({
        "role": "user",
        "content": content
    })

    response = await openai_client.chat.completions.create(
        messages=messages,
        model=MODEL
    )
    return response.choices[0].message.content

async def send_response(update: Update, reply_text: str) -> None:
    """Отправляет ответ пользователю, разбивая длинные сообщения на части."""
    max_length = 4096
    reply_chunks = [reply_text[i:i+max_length] for i in range(0, len(reply_text), max_length)]
    for chunk in reply_chunks:
        await update.message.reply_text(chunk, parse_mode=ParseMode.MARKDOWN)

async def handle_message(update: Update, context) -> None:
    """Обрабатывает входящие сообщения от пользователей."""
    try:
        chat_id = update.effective_chat.id
        user_id = update.effective_user.id

        logger.info(
            f"""
        ====== Новое сообщение ======
        Chat ID: {chat_id}
        User ID: {user_id}
        Has Photo: {bool(update.message.photo)}
        Caption: {update.message.caption}
        Text: {update.message.text}
        ===========================
        """
        )

        if not await is_allowed_chat(chat_id):
            return

        user_status = (await update.effective_chat.get_member(user_id)).status
        user_is_admin = await is_admin(user_status)

        # Определяем текст сообщения (из caption если есть фото, иначе из text)
        message_text = (
            update.message.caption
            if update.message.photo
            else update.message.text or ""
        )
        photos = update.message.photo

        logger.info(f"Обработка сообщения: {message_text}")
        logger.info(f"Наличие фото: {bool(photos)}")

        if not message_text.startswith(f"@{context.bot.username}"):
            logger.info("Сообщение не начинается с обращения к боту")
            return

            message_text = message_text.replace(f"@{context.bot.username}", "").strip()

        if not await check_message_limit(chat_id, user_id, user_is_admin):
            await update.message.reply_text("Вы превысили лимит сообщений на сегодня.")
            return

        # Подготовка контента для API
        content, image_paths = await prepare_content_for_api(message_text, photos)

        # Получение контекста и ответа
        context_messages = await get_user_context(chat_id, user_id)
        reply_text = await get_ai_response(context_messages, content)

        # Отправка ответа и обновление контекста
        await send_response(update, reply_text)
        await update_user_context(chat_id, user_id, reply_text, "assistant")

        # Очистка временных файлов
        if image_paths:
            await cleanup_temp_images(image_paths)

    except Exception as e:
        logger.exception(f"Ошибка при обработке сообщения: {e}")


async def reset_message_counters() -> None:
    """Сбрасывает счетчики сообщений каждые сутки."""
    while True:
        global message_counters
        message_counters = {}
        logger.info("Счетчики сообщений сброшены.")
        await asyncio.sleep(24 * 60 * 60)


async def main() -> None:
    """Основная функция для запуска бота."""
    try:
        await load_chat_logs()

        application = Application.builder().token(BOT_TOKEN).build()

        # Инициализируем бота и получаем его username
        await application.initialize()
        bot_username = (await application.bot.get_me()).username

        # Теперь используем bot_username в фильтрах
        application.add_handler(
            MessageHandler(
                (filters.PHOTO & filters.CaptionRegex(f"^@{bot_username}"))
                | (filters.TEXT & filters.Regex(f"^@{bot_username}")),
                handle_message,
            )
        )

        logger.info("Запуск бота...")
        await application.start()
        await application.updater.start_polling()

        asyncio.create_task(reset_message_counters())
    except Exception as e:
        logger.exception(f"Ошибка при запуске бота: {e}")

if __name__ == "__main__":
    main()
