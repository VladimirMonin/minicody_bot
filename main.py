"""
Модуль Telegram-бота для поддержки студентов в группах.
Бот поддерживает асинхронную работу, ограничивает количество обращений студентов в сутки и обращений к OpenAI API.
Контекст общения сохраняется в JSON формате.
Модуль использует принципы единой ответственности (SRP) и избегает повторений кода (DRY).
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional
from telegram import Update, Bot
from telegram.ext import Application, CommandHandler, MessageHandler, filters
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv


# Загружаем переменные окружения из файла .env
load_dotenv()


# Чтение переменных окружения из файла .env
ALLOWED_CHATS = os.getenv("ALLOWED_CHATS")
MAX_MESSAGES_PER_DAY = os.getenv("MAX_MESSAGES_PER_DAY")
CONTEXT_EXPIRATION_MINUTES = os.getenv("CONTEXT_EXPIRATION_MINUTES")
CONTEXT_MESSAGE_LIMIT = os.getenv("CONTEXT_MESSAGE_LIMIT")
OPENAI_API_LIMIT_PER_SECOND = os.getenv("OPENAI_API_LIMIT_PER_SECOND")
JSON_LOG_FILE = os.getenv("JSON_LOG_FILE")
OPEN_AI_API_KEY = os.getenv("OPEN_AI_API_KEY")