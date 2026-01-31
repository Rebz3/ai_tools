import argparse
import gzip
import os
import re
import tarfile
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

TOP_N = 10
MAX_LINE_LENGTH = 8000  # макс. длина строки


def format_seconds(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(round(seconds % 60))
    return f"{h:02}:{m:02}:{s:02}"


def now_human() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def debug_print(msg: str, enabled: bool):
    if enabled:
        print("DEBUG:", msg)


def trim_long_line(line: str) -> str:
    if len(line) > MAX_LINE_LENGTH:
        return line[:MAX_LINE_LENGTH] + f"[ ... Обрезано: {len(line) - MAX_LINE_LENGTH} символов ... ]"
    return line


def load_llm_config(env_path=None) -> dict:
    if env_path is None:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(env_path)
    cfg = {
        "LLM_BASE_URL": os.getenv("LLM_BASE_URL"),
        "LLM_API_KEY": os.getenv("LLM_API_KEY"),
        "LLM_MODEL_NAME": os.getenv("LLM_MODEL_NAME"),
        "LLM_TEMPERATURE": os.getenv("LLM_TEMPERATURE"),
        "LLM_TOP_P": os.getenv("LLM_TOP_P"),
        "LLM_TOP_K": os.getenv("LLM_TOP_K"),
        "LLM_MAX_TOKENS": os.getenv("LLM_MAX_TOKENS"),
    }
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise RuntimeError(f"Не найдены переменные в .env: {', '.join(missing)}")
    cfg["LLM_TEMPERATURE"] = float(cfg["LLM_TEMPERATURE"])
    cfg["LLM_TOP_P"] = float(cfg["LLM_TOP_P"])
    cfg["LLM_TOP_K"] = int(cfg["LLM_TOP_K"])
    cfg["LLM_MAX_TOKENS"] = int(cfg["LLM_MAX_TOKENS"])
    return cfg


def collect_log_files(input_path: str, debug: bool) -> list[str]:
    allowed_exts = (".tar.gz", ".tgz", ".tar", ".gz", ".log")
    all_files = []
    if os.path.isfile(input_path) and input_path.endswith(allowed_exts):
        all_files = [input_path]
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.endswith(allowed_exts):
                    all_files.append(os.path.join(root, f))
    else:
        raise Exception(f"Путь {input_path} не является файлом или директорией")
    debug_print(f"Собрано {len(all_files)} файлов для анализа", debug)
    return sorted(all_files)


def extract_log_lines_extended(file_path: str, debug: bool) -> list[dict]:
    log_lines = []
    base_name = os.path.basename(file_path)
    try:
        if file_path.endswith((".tar.gz", ".tgz")):
            debug_print(f"Читаем tar.gz архив: {file_path}", debug)
            with tarfile.open(file_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".log"):
                        f = tar.extractfile(member)
                        if f:
                            for line in f:
                                try:
                                    decoded_line = trim_long_line(line.decode("utf-8", errors="ignore").rstrip("\n"))
                                    log_lines.append(
                                        {"archive": base_name, "logfile": member.name, "line": decoded_line}
                                    )
                                except Exception:
                                    continue
        elif file_path.endswith(".tar"):
            debug_print(f"Читаем tar архив: {file_path}", debug)
            with tarfile.open(file_path, "r:") as tar:
                for member in tar.getmembers():
                    if member.isfile() and member.name.endswith(".log"):
                        f = tar.extractfile(member)
                        if f:
                            for line in f:
                                try:
                                    decoded_line = trim_long_line(line.decode("utf-8", errors="ignore").rstrip("\n"))
                                    log_lines.append(
                                        {"archive": base_name, "logfile": member.name, "line": decoded_line}
                                    )
                                except Exception:
                                    continue
        elif file_path.endswith(".gz"):
            debug_print(f"Читаем gzip файл: {file_path}", debug)
            with gzip.open(file_path, "rt", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    log_lines.append(
                        {"archive": base_name, "logfile": base_name, "line": trim_long_line(line.rstrip("\n"))}
                    )
        elif file_path.endswith(".log"):
            debug_print(f"Читаем обычный лог-файл: {file_path}", debug)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    log_lines.append({"archive": "", "logfile": base_name, "line": trim_long_line(line.rstrip("\n"))})
        else:
            print(f"[WARN] Файл игнорируется (неподдерживаемый формат): {file_path}")
    except Exception as e:
        print(f"[ERROR] Ошибка при обработке файла {file_path}: {e}")
    return log_lines


def parse_timestamp(log_line: str):
    patterns = [
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})?)",
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)",
        r"(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2} [+\-]\d{4})",
    ]
    for pat in patterns:
        m = re.search(pat, log_line)
        if m:
            ts_str = m.group(1)
            for fmt in ("%Y-%m-%d %H:%M:%S,%f", "%Y-%m-%dT%H:%M:%SZ", "%d/%b/%Y:%H:%M:%S %z"):
                try:
                    return datetime.strptime(ts_str, fmt)
                except Exception:
                    continue
    return None


def find_nonstandard_log_entries(all_log_entries: list[dict]) -> Counter:
    counter = Counter()
    for entry in all_log_entries:
        line = entry["line"].strip()
        if line and parse_timestamp(line) is None:
            counter[line] += 1
    return counter


def analyze_logs(all_log_entries: list[dict]) -> dict:
    err_kw = "error"
    panic_kw = "panic:"
    info_kw = "info"
    dbg_kw = "debug"
    reconnect_kws = ["reconnect", "disconnected", "connected"]
    oom_kw = ["out of memory"]
    stack_pat = re.compile(r"(traceback|stack trace|stacktrace|at .+\(.+\))", re.IGNORECASE)

    err_pf = defaultdict(lambda: defaultdict(Counter))
    all_err = Counter()
    all_msg = Counter()
    info_msg = Counter()
    debug_msg = Counter()
    reconn_msg = Counter()
    panics = []
    stacks = []
    critical = []
    out_of_memory = []

    for e in all_log_entries:
        archive, logfile, line = e["archive"], e["logfile"], e["line"]
        sl = line.strip()
        all_msg[sl] += 1
        low = line.lower()

        if err_kw in low:
            err_pf[archive][logfile][sl] += 1
            all_err[sl] += 1
            ts = parse_timestamp(line)
            if ts:
                critical.append((ts, archive, logfile, sl))
        if info_kw in low:
            info_msg[sl] += 1
        if dbg_kw in low:
            debug_msg[sl] += 1
        if any(k in low for k in reconnect_kws):
            reconn_msg[sl] += 1
        if panic_kw in low:
            panics.append({"archive": archive, "logfile": logfile, "line": sl})
            ts = parse_timestamp(line)
            if ts:
                critical.append((ts, archive, logfile, sl))
        if any(oom_kw in low for oom_kw in oom_kw):
            out_of_memory.append({"archive": archive, "logfile": logfile, "line": sl})
            ts = parse_timestamp(line)
            if ts:
                critical.append((ts, archive, logfile, sl))
        if stack_pat.search(line):
            stacks.append({"archive": archive, "logfile": logfile, "line": sl})

    critical.sort(key=lambda x: x[0])
    return {
        "error_counters_per_file": err_pf,
        "all_errors_counter": all_err,
        "all_messages_counter": all_msg,
        "info_messages_counter": info_msg,
        "debug_messages_counter": debug_msg,
        "reconnect_events_counter": reconn_msg,
        "panics": panics,
        "stack_traces": stacks,
        "critical_errors_by_time": critical,
        "out_of_memory": out_of_memory,
    }


def read_prompt_template(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл шаблона промпта не найден: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def make_summary(an: dict, top_n: int, prompt_template: str, nonstd_counter: Counter) -> str:
    per_file_block = []
    for archive, files_dict in an["error_counters_per_file"].items():
        if archive:
            per_file_block.append(f"Архив: {archive}")
            for logfile, counter in files_dict.items():
                per_file_block.append(f" Файл: {logfile}")
                for msg, count in counter.most_common(top_n):
                    per_file_block.append(f" [{count} раз] {msg}")
        else:
            for logfile, counter in files_dict.items():
                per_file_block.append("")
                per_file_block.append(f"Лог файл: {logfile}")
                for msg, count in counter.most_common(top_n):
                    per_file_block.append(f" [{count} раз] {msg}")
        per_file_block.append("")
    nonstd_block = (
        "\n".join(f"[{count} раз] {line}" for line, count in nonstd_counter.most_common(top_n))
        if nonstd_counter
        else "Нет строк нестандартного формата"
    )
    oom_lines = (
        "\n".join(f"{p['archive']} | {p['logfile']} | {p['line']}" for p in an.get("out_of_memory", [])[:top_n])
        or "Нет out of memory (OOM) ошибок"
    )
    return prompt_template.format(
        top_errors="\n".join(f"[{c} раз] {m}" for m, c in an["all_errors_counter"].most_common(top_n)) or "Нет ошибок",
        per_file="\n".join(per_file_block),
        top_msgs="\n".join(f"[{c} раз] {m}" for m, c in an["all_messages_counter"].most_common(top_n)),
        top_info="\n".join(f"[{c} раз] {m}" for m, c in an["info_messages_counter"].most_common(top_n)),
        top_debug="\n".join(f"[{c} раз] {m}" for m, c in an["debug_messages_counter"].most_common(top_n)),
        reconnect_info="\n".join(f"[{c} раз] {m}" for m, c in an["reconnect_events_counter"].most_common(top_n))
        or "Нет событий",
        critical="\n".join(
            f"{ts.isoformat()} | {a} | {lf} | {m}" for ts, a, lf, m in an["critical_errors_by_time"][: 4 * top_n]
        ),
        panic="\n".join(f"{p['archive']} | {p['logfile']} | {p['line']}" for p in an["panics"][:top_n]),
        stack="\n".join(f"{s['archive']} | {s['logfile']} | {s['line']}" for s in an["stack_traces"][:top_n]),
        oom=oom_lines,
        nonstandard_lines=nonstd_block,
    )


def get_llm_report(config: dict, prompt: str) -> str:
    """Отправляет запрос в LLM и возвращает результат."""
    try:
        url = f"{config['LLM_BASE_URL'].rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {config['LLM_API_KEY']}", "Content-Type": "application/json"}
        payload = {
            "model": config["LLM_MODEL_NAME"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config["LLM_TEMPERATURE"],
            "top_p": config["LLM_TOP_P"],
            "top_k": config["LLM_TOP_K"],
            "max_tokens": config["LLM_MAX_TOKENS"],
        }
        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        try:
            content = data["choices"][0]["message"]["content"]
            if content:
                return content
        except (KeyError, IndexError, TypeError):
            pass

        try:
            content = data["choices"][0]["text"]
            if content:
                return content
        except (KeyError, IndexError, TypeError):
            pass

        raise RuntimeError(f"Неверный формат ответа LLM API: {data}")
    except Exception as e:
        print(f"[ERROR] Ошибка при обращении к LLM: {e}")
        return ""


def save_ai_response(
    content: str, tag: str, meta_info: dict, llm_config: dict, prompt_template_path: str, raw_prompt_data: str
):
    dir_out = Path(__file__).parents[2] / "_out"
    core_dump_folder = Path(__file__).resolve().parents[2] / "_out" / "coredumps"
    dir_out.mkdir(parents=True, exist_ok=True)
    out_path = (dir_out / f"{tag}.html").resolve()
    core_dump_files = []
    if core_dump_folder.is_dir():
        core_dump_files = [str(p.name) for p in core_dump_folder.iterdir()]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write('<html><head><meta charset="UTF-8"></head><body style="margin:5%;">\n')
        f.write("<h2>Meta Information</h2><ul>")
        f.write(f"<li>Generation time: {now_human()}</li>")
        for k, v in (meta_info or {}).items():
            f.write(f"<li>{k}: {v}</li>")
        if llm_config:
            f.write("<li>LLM Parameters<ul>")
            for k in ["LLM_MODEL_NAME", "LLM_TEMPERATURE", "LLM_TOP_P", "LLM_TOP_K", "LLM_MAX_TOKENS"]:
                f.write(f"<li>{k}: {llm_config.get(k, '-')}</li>")
            f.write("</ul></li>")
        f.write(f"<li>Prompt template path: {prompt_template_path}</li></ul><hr>")
        if core_dump_files:
            f.write(f"<li>Найдены coredump файлы агентов: {str(core_dump_files)}</li></ul><hr>")
        f.write(content)
        if raw_prompt_data:
            f.write("<hr style='margin-top:40px; border: 2px solid #444;'>")
            f.write("<div style='background:#f0f0f0; padding:15px; margin-top:10px;'>")
            f.write("<h3 style='margin-top:0;'>RAW PROMPT DATA (Данные, переданные в AI)</h3>")
            f.write(
                "<p style='font-size:90%;color:#555;'>Ниже представлен полный текст запроса, который был отправлен в LLM для анализа. Этот блок служебный и не является частью финального отчёта.</p>"
            )
            f.write("<pre style='white-space: pre-wrap; font-size:90%;'>")
            f.write(raw_prompt_data)
            f.write("</pre></div>")
        f.write("</body></html>")
    return out_path


def main():
    """
    AI-анализатор логов с использованием LLM.

    Что делает:
    - Извлекает строки из логов и архивов (.tar, .tar.gz, .gz, .log).
    - Находит ошибки, паники, повторяющиеся сообщения, события reconnect, стек-трейсы.
    - Вычисляет топ-N строк нестандартного формата (без таймстампа).
    - Формирует статистику и отправляет в LLM для генерации HTML-отчёта.
    - Сохраняет результат в `_out` (HTML с метаинформацией и блоком RAW PROMPT DATA).

    Куда сохраняет:
    - `_out/logs_ai_analysis.html` в корне проекта.

    Сценарии использования:
    1) Анализ одного архива:
       python ai_log_analyzer.py /path/to/archive.tar.gz

    2) Анализ папки со множеством архивов и логов:
       python ai_log_analyzer.py path/to/logs_directory/

    3) Использование кастомного шаблона промпта:
       python ai_log_analyzer.py path/to/logs/ --prompt_template path/to/custom_prompt.txt

    4) Указание альтернативного .env файла с параметрами LLM:
       python ai_log_analyzer.py path/to/logs/ --env path/to/custom.env

    5) Включение подробных отладочных сообщений:
       python ai_log_analyzer.py path/to/logs/ --debug

    По результату в директории _out рядом со скриптом появляется файл отчёта в формате Markdown.
    """
    parser = argparse.ArgumentParser(description="AI-анализатор логов")
    parser.add_argument("input_path")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--env", default=os.path.join(script_dir, ".env"))
    parser.add_argument("--prompt_template", default=os.path.join(script_dir, "logs_prompt_template.txt"))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    debug_enabled = args.debug
    debug_print(f"Input path: {args.input_path}", debug_enabled)
    debug_print(f".env path: {args.env}", debug_enabled)
    debug_print(f"Prompt template path: {args.prompt_template}", debug_enabled)

    t0_all = time.time()
    prompt_template = read_prompt_template(args.prompt_template)

    log_files = collect_log_files(args.input_path, debug_enabled)

    t0 = time.time()
    all_log_entries = []
    for path in log_files:
        lines = extract_log_lines_extended(path, debug_enabled)
        debug_print(f"Обработан файл: {os.path.basename(path)}. Извлечено строк: {len(lines)}", debug_enabled)
        all_log_entries.extend(lines)
    print(f"Время извлечения всех логов: {format_seconds(time.time() - t0)}")
    if not all_log_entries:
        print("Нет данных для анализа.")
        return

    nonstd_counter = find_nonstandard_log_entries(all_log_entries)
    if debug_enabled:
        print(f"\nНайдено уникальных нестандартных строк: {len(nonstd_counter)}")
        for line, cnt in nonstd_counter.most_common(20):
            print(f"[{cnt} раз] {line}")
        if len(nonstd_counter) > 20:
            print(f"... и ещё {len(nonstd_counter) - 20} уникальных строк")

    t0 = time.time()
    analysis = analyze_logs(all_log_entries)
    print(f"Время анализа логов: {format_seconds(time.time() - t0)}")

    config = load_llm_config(args.env)
    summary = make_summary(analysis, TOP_N, prompt_template, nonstd_counter)
    print("Данные переданы в AI для анализа...")
    t0 = time.time()
    ai_report = get_llm_report(config, summary)
    print(f"Время AI-анализа: {format_seconds(time.time() - t0)}")

    out_path = save_ai_response(
        ai_report,
        "logs_ai_analysis",
        {"Файлов обработано": len(log_files), "Всего строк": len(all_log_entries)},
        config,
        args.prompt_template,
        summary,
    )

    print("\nAI-анализ логов завершён!")
    print(f"Результат сохранён в файле: {out_path}")
    print(f"Общее время выполнения: {format_seconds(time.time() - t0_all)}")


if __name__ == "__main__":
    main()
