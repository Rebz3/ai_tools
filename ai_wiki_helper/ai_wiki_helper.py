import argparse
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Set

import requests
from dotenv import load_dotenv

# Список префиксов для поиска релевантных ссылок
RELEVANT_LINK_PREFIXES = ["[SR]", "[SRS]"]


def get_script_dir():
    return os.path.dirname(os.path.abspath(__file__))


SCRIPT_DIR = get_script_dir()
OUT_PATH = None


def abspath_from_script_dir(*path_parts):
    return os.path.join(SCRIPT_DIR, *path_parts)


def format_seconds(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(round(seconds % 60))
    return f"{hours:02}:{minutes:02}:{secs:02}"

def now_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


load_dotenv(abspath_from_script_dir(".env"))

CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_TOKEN = os.getenv("CONFLUENCE_TOKEN")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")
LLM_TEMPERATURE = os.getenv("LLM_TEMPERATURE")
LLM_TOP_P = os.getenv("LLM_TOP_P")
LLM_TOP_K = os.getenv("LLM_TOP_K")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS") or 8000)

if not CONFLUENCE_TOKEN:
    print("Ошибка: не заполнен токен Confluence")
    sys.exit(1)
for var in [CONFLUENCE_URL, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_NAME]:
    if not var:
        raise Exception("Ошибка: обязательная переменная окружения не заполнена!")


def get_confluence_page_html(page_id: str) -> dict:
    url = f"{CONFLUENCE_URL}/rest/api/content/{page_id}"
    headers = {
        "Authorization": f"Bearer {CONFLUENCE_TOKEN}",
        "Accept": "application/json",
    }
    params = {"expand": "body.view,body.storage"}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    return {"title": data["title"], "html": data["body"]["view"]["value"], "markup": data["body"]["storage"]["value"]}


def find_relevant_links(html: str) -> Set[str]:
    link_ids = set()
    # Поддержка обоих форматов URL: с pageId= в query-параметрах и в пути
    patterns = [
        r'<a [^>]*href="[^"]*\?pageId=(\d+)[^"]*"[^>]*>(.*?)</a>',
        r'<a [^>]*href="[^"]*/pages/(\d+)[^"]*"[^>]*>(.*?)</a>',
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, html, re.IGNORECASE):
            page_id, text = match.groups()
            if any(text.upper().startswith(prefix) for prefix in RELEVANT_LINK_PREFIXES):
                link_ids.add(page_id)
    return link_ids


def recursive_collect_pages(page_id: str, max_depth: int) -> Dict[str, Dict]:
    collected = {}
    queue = deque()
    visited = set()
    queue.append((page_id, 0))
    order = []
    while queue:
        cur_id, depth = queue.popleft()
        if cur_id in visited or depth > max_depth:
            continue
        data = get_confluence_page_html(cur_id)
        collected[cur_id] = data
        order.append((cur_id, data["title"], depth))
        visited.add(cur_id)
        if depth < max_depth:
            links = find_relevant_links(data["html"])
            for link_id in links:
                if link_id not in visited:
                    queue.append((link_id, depth + 1))
    collected["_order"] = order
    return collected


def get_checklist_content() -> str:
    checklist_id = "433137998"
    return get_confluence_page_html(checklist_id)["markup"]


def load_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        raise Exception(f"Шаблон промпта {path} не найден")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_text_for_chunks(text: str, max_tokens: int, approx_char_per_token=4) -> List[str]:
    max_len = max_tokens * approx_char_per_token
    paragraphs = text.split("\n\n")
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_len:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def make_summarize_prompt_template():
    return (
        "Вот часть требований из документа:\n\n"
        "{chunk}\n\n"
        "Сделайте краткий (50-200 слов) консолидированный пересказ сути этого участка требований, сохранив важные SR-ссылки и ограничения. Не упускайте важные детали."
    )


def summarize_chunk(chunk: str, summarize_prompt_template: str) -> str:
    prompt = summarize_prompt_template.replace("{chunk}", chunk)
    url = f"{LLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(LLM_TEMPERATURE),
        "top_p": float(LLM_TOP_P),
        "top_k": int(LLM_TOP_K),
        "max_tokens": LLM_MAX_TOKENS // 4,
        "response_format": {"type": "text"},
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def process_all_chunks_and_summarize(pages: Dict[str, Dict], max_tokens_per_chunk=2000, dry_run=False):
    all_chunks = []
    for pid, data in pages.items():
        if pid == "_order":
            continue
        page_chunks = split_text_for_chunks(data["markup"], max_tokens=max_tokens_per_chunk)
        all_chunks.extend(page_chunks)
    summarize_prompt = make_summarize_prompt_template()
    summaries = []
    print(f"Будет обработано {len(all_chunks)} чанков для свёртки.")

    start_time = time.time()
    if dry_run:
        # Мок-режим: не отправляем запросы к ИИ и не выводим чанки!
        for idx, chunk in enumerate(all_chunks, 1):
            summaries.append(f"[Мок-суммаризация чанка {idx}]")
        elapsed = time.time() - start_time
        return summaries, elapsed
    else:
        for idx, chunk in enumerate(all_chunks, 1):
            print(f"Обработка чанка {idx}/{len(all_chunks)} для суммаризации...")
            summary = summarize_chunk(chunk, summarize_prompt)
            summaries.append(summary.strip())
        elapsed = time.time() - start_time
        return summaries, elapsed


def build_full_prompt(template: str, chunk_summaries: List[str], checklist: str | None = None):
    all_requirements = "\n\n".join(f"---\n{summary}\n---" for summary in chunk_summaries)
    filled = template.replace("{requirements}", all_requirements)
    if "{checklist}" in filled:
        if checklist is not None:
            filled = filled.replace("{checklist}", checklist)
        else:
            filled = filled.replace("{checklist}", "")
    return filled


def generate_ai_completion(prompt: str) -> str:
    url = f"{LLM_BASE_URL}/chat/completions"
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(LLM_TEMPERATURE),
        "top_p": float(LLM_TOP_P),
        "top_k": int(LLM_TOP_K),
        "max_tokens": int(LLM_MAX_TOKENS),
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def save_response(
    content: str, tag: str, suffix: str = "", meta_info: dict = None, prompt_template_path: str = None
):
    if OUT_PATH:
        # Пользователь задал путь — используем его
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        out_path = OUT_PATH
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = abspath_from_script_dir("_out")
        os.makedirs(out_dir, exist_ok=True)
        safe_tag = "".join(c for c in tag if c.isalnum() or c in (" ", "_", "-")).rstrip()
        out_path = os.path.join(out_dir, f"{timestamp}_{safe_tag}{suffix}.html")

    meta_lines = []
    if meta_info:
        meta_lines.append(f"<li>Page ID: {meta_info.get('page_id', '-')}</li>")
        meta_lines.append(f"<li>Page Title: {meta_info.get('page_title', '-')}</li>")
        meta_lines.append("<li>LLM Parameters:</li>")
        meta_lines.append(f"<li>Model: {meta_info.get('model', '-')}</li>")
        meta_lines.append(f"<li>Temperature: {meta_info.get('temperature', '-')}</li>")
        meta_lines.append(f"<li>Top K: {meta_info.get('top_k', '-')}</li>")
        meta_lines.append(f"<li>Top P: {meta_info.get('top_p', '-')}</li>")
        meta_lines.append(f"<li>Max Tokens: {meta_info.get('max_tokens', '-')}</li>")

    if prompt_template_path:
        meta_lines.append(f"<li>Prompt template path: {prompt_template_path}</li>")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<html><head><meta charset="UTF-8"></head><body style="margin:5%;">\n')
        f.write("<h2>Meta Information</h2><ul>")
        f.write(f"<li>Generation time: {now_human()}</li>")
        if meta_lines:
            f.write("\n".join(meta_lines) + "\n")
        f.write(content)

    print(f"Ответ ИИ сохранён в: {out_path}")


def print_relevant_titles(pages: Dict[str, Dict]):
    if "_order" not in pages:
        return
    print("Обход релевантных страниц (по глубинам):")
    for pid, title, depth in pages["_order"]:
        print(f"  Глубина {depth}: {title} (pageId={pid})")

def main():
    """
    Скрипт для сбора и анализа требований из Confluence с помощью LLM.
    --dry_run: просмотреть структуру обработки и параметры, но не отправлять никаких запросов к ИИ и не сохранять результат.
    """
    parser = argparse.ArgumentParser(description="Обход требований Confluence с поэтапной свёрткой для промптов.")
    parser.add_argument("--page_id", required=True, help="ID страницы Confluence для начала обхода")
    parser.add_argument("--depth", type=int, default=0, help="Глубина обхода релевантных ссылок >=0 (по умолчанию 0)")
    parser.add_argument("--role", default="qa", choices=["qa", "analyst"], help="Роль: qa (по умолчанию) или analyst")
    parser.add_argument(
        "--chunk_size", type=int, default=2000, help="Максимальное число токенов на один чанк (по умолчанию 2000)"
    )
    parser.add_argument(
        "--prompt_template",
        default=None,
        help="Путь к кастомному шаблону промпта (если задан, переопределяет выбор по роли)",
    )
    parser.add_argument(
        "--out_path",
        help="Путь к выходному файлу (необязательно: если не задан, будет автоматически создан в папке _out/)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Мок-режим: не отправлять никаких запросов и не сохранять результат, только показать структуру и параметры",
    )
    args = parser.parse_args()
    global OUT_PATH
    OUT_PATH = args.out_path

    print(f"Сбор страниц начиная с {args.page_id} с глубиной обхода {args.depth}...")
    t0_all = time.time()

    if args.depth == 0:
        print("Глубина 0, загружаем одну страницу целиком без чанков...")
        page_data = get_confluence_page_html(args.page_id)
        pages = {args.page_id: page_data}
        chunk_summaries = None
        chunk_time = 0.0
    else:
        pages = recursive_collect_pages(args.page_id, args.depth)
        if args.depth > 0:
            print_relevant_titles(pages)
        chunk_summaries = None
        chunk_time = 0.0

    if args.prompt_template is not None:
        prompt_template_path = args.prompt_template
        if not os.path.isabs(prompt_template_path):
            prompt_template_path = abspath_from_script_dir(prompt_template_path)
        print(f"Используется пользовательский шаблон промпта: {prompt_template_path}")
    else:
        prompt_template_path = abspath_from_script_dir(f"{args.role}_prompt_template.txt")
        print(f"Используем шаблон промпта по роли {args.role}: {prompt_template_path}")

    prompt_template = load_prompt_template(prompt_template_path)

    if "{checklist}" in prompt_template:
        print("Шаблон содержит {checklist}, загружаем чеклист...")
        checklist = get_checklist_content()
        print("Чеклист загружен.")
    else:
        checklist = None

    if args.depth == 0:
        print("Формируем запрос из полного содержания страницы...")
        full_prompt = prompt_template.replace("{requirements}", pages[args.page_id]["markup"])
        if checklist is not None:
            full_prompt = full_prompt.replace("{checklist}", checklist)
        else:
            full_prompt = full_prompt.replace("{checklist}", "")
    else:
        print("Делим текст требований на чанки и суммируем...")
        chunk_summaries, chunk_time = process_all_chunks_and_summarize(pages, args.chunk_size, dry_run=args.dry_run)
        print(f"Время суммаризации чанков: {format_seconds(chunk_time)}")
        print("Формируем итоговый запрос к ИИ...")
        full_prompt = build_full_prompt(prompt_template, chunk_summaries, checklist)

    if args.dry_run:
        print("\n===== DRY-RUN РЕЖИМ (мок-режим, без отправки и сохранения) =====")
        print(f"Роль: {args.role}")
        print(f"Глубина обхода: {args.depth}")
        print(f"Шаблон промпта: {prompt_template_path}")
        print(f"Количество собранных страниц: {len([k for k in pages if k != '_order'])}")
        if args.depth > 0 and "_order" in pages:
            print("Заголовки обработанных релевантных страниц (по глубине):")
            for pid, title, depth in pages["_order"]:
                print(f"  Глубина {depth}: {title} (pageId={pid})")
        if chunk_summaries is not None:
            print(f"Количество чанков для суммаризации: {len(chunk_summaries)}")
            print(f"Время мок-суммаризации чанков: {format_seconds(chunk_time)}")
        else:
            print("Суммаризация чанков не используется.")
        string_limit = 6000
        print(f"\nПервые {string_limit} символов сформированного промпта:\n{'-' * 40}\n{full_prompt[:string_limit]}\n{'-' * 40}")
        print("\nЗапросы к ИИ не выполнялись, файл не сохранён (dry-run).")
    else:
        print("Отправляем запрос к ИИ для анализа...")
        t_ai0 = time.time()
        ai_response = generate_ai_completion(full_prompt)
        t_ai1 = time.time()
        ai_time = t_ai1 - t_ai0
        print(f"Время ответа ИИ: {format_seconds(ai_time)}")
        main_title = pages[args.page_id]["title"]
        if args.prompt_template is not None:
            suffix = ""  # без суффикса
        else:
            suffix = "_test_cases" if args.role == "qa" else "_analysis"
        meta_info = {
            "page_id": args.page_id,
            "page_title": pages[args.page_id]["title"],
            "model": LLM_MODEL_NAME,
            "temperature": LLM_TEMPERATURE,
            "top_k": LLM_TOP_K,
            "top_p": LLM_TOP_P,
            "max_tokens": LLM_MAX_TOKENS,
        }
        save_response(ai_response, main_title, suffix, meta_info, prompt_template_path)
        save_response(ai_response, main_title, suffix, meta_info, prompt_template_path)

        print(f"Всё выполнено за {format_seconds(time.time() - t0_all)}")


if __name__ == "__main__":
    main()
