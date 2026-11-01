# -*- coding: utf-8 -*-
"""ch10 · 工具调用智能体：让模型用三个本地工具完成五道题（DeepSeek function calling）。

课程原 notebook 用 OpenAI 的 function calling 搭智能体；本机可用 DeepSeek 的同一套协议
（`tools` 参数 + `tool_calls` 回传），故改为 DeepSeek + 本地工具。

工具：calculator（四则与幂运算）、count_words（读本地文本并数词）、celsius_to_fahrenheit。
指标：五题的正确数、工具调用次数、对话轮数；全部过程写入 results/trace.json。

运行前设置：$env:DEEPSEEK_API_KEY="sk-..."
"""
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

BASE = "https://api.deepseek.com"
MODEL = "deepseek-chat"
KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
HERE = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(HERE, "data", "sample.txt")
WORDS_DOC = ("A robot that works outside a lab has to carry its own power, sense its surroundings, "
             "and decide what to do when a part fails. These three requirements shape every board "
             "and every line of code that goes into it.")


def ensure_data():
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            f.write(WORDS_DOC + "\n")


def calculator(expression):
    if not re.fullmatch(r"[\d\s\.\+\-\*\/\(\)\^%]+", expression):
        return {"error": "only arithmetic is allowed"}
    expr = expression.replace("^", "**")
    try:
        value = eval(expr, {"__builtins__": {}}, {"math": math})  # 受限于上面的字符白名单
    except Exception as e:
        return {"error": "%s: %s" % (type(e).__name__, e)}
    return {"value": value}


def count_words(path):
    if os.path.basename(path) != "sample.txt":
        return {"error": "only sample.txt is readable"}
    with open(DATA_FILE, encoding="utf-8") as f:
        text = f.read()
    return {"words": len(text.split()), "characters": len(text)}


def celsius_to_fahrenheit(celsius):
    return {"fahrenheit": float(celsius) * 9 / 5 + 32}


TOOLS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression. Use for any numeric computation.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. '0.175*240' or '2**10+24'"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "count_words",
        "description": "Count words in the local file data/sample.txt.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "always 'sample.txt'"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "celsius_to_fahrenheit",
        "description": "Convert a Celsius temperature to Fahrenheit.",
        "parameters": {"type": "object", "properties": {
            "celsius": {"type": "number"}},
            "required": ["celsius"]}}},
]

REGISTRY = {"calculator": calculator, "count_words": count_words, "celsius_to_fahrenheit": celsius_to_fahrenheit}

TASKS = [
    {"q": "What is 17.5% of 240? Use the calculator.", "expect": 42.0},
    {"q": "How many words are in data/sample.txt?", "expect": 40},
    {"q": "Convert 37.5 degrees Celsius to Fahrenheit.", "expect": 99.5},
    {"q": "Take the word count of data/sample.txt, multiply it by 12, and add 3.", "expect": 483},
    {"q": "What is 2 to the power of 10, plus 24?", "expect": 1048},
]


def call_api(messages, retries=3):
    body = json.dumps({"model": MODEL, "messages": messages, "tools": TOOLS, "temperature": 0,
                       "max_tokens": 512}).encode("utf-8")
    req = urllib.request.Request(BASE + "/chat/completions", data=body,
                                 headers={"Authorization": "Bearer " + KEY, "Content-Type": "application/json"})
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = "HTTP %s %s" % (e.code, e.read()[:160].decode("utf-8", "replace"))
        except Exception as e:
            last = "%s %s" % (type(e).__name__, str(e)[:120])
        time.sleep(1.5 * (i + 1))
    raise RuntimeError(last)


def run_task(task, max_rounds=6):
    messages = [
        {"role": "system", "content": "You are an engineering assistant. Use the tools when a calculation "
                                      "or a file read is needed, then give the final numeric answer."},
        {"role": "user", "content": task["q"]},
    ]
    tool_calls, trace = 0, []
    for _ in range(max_rounds):
        resp = call_api(messages)
        msg = resp["choices"][0]["message"]
        messages.append(msg)
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        for c in calls:
            name = c["function"]["name"]
            args = json.loads(c["function"]["arguments"] or "{}")
            result = REGISTRY[name](**args) if name in REGISTRY else {"error": "unknown tool"}
            tool_calls += 1
            trace.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result)})
    final = messages[-1].get("content") or ""
    core = re.sub(r"\([^)]*\)", " ", final).replace(",", "")
    nums = re.findall(r"-?\d+(?:\.\d+)?", core)
    got = float(nums[-1]) if nums else None
    ok = got is not None and abs(got - float(task["expect"])) < 1e-6
    return {"question": task["q"], "expect": task["expect"], "answer": final.strip()[:200],
            "parsed": got, "correct": bool(ok), "tool_calls": tool_calls,
            "rounds": len([m for m in messages if m.get("role") == "assistant"]), "trace": trace}


def main():
    if not KEY:
        print("缺少 DEEPSEEK_API_KEY 环境变量", file=sys.stderr)
        return 2
    ensure_data()
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    log_path = os.path.join(HERE, "results", "log.txt")
    results = []
    for t in TASKS:
        r = run_task(t)
        results.append(r)
        line = "[%s] %s -> %s (expect %s) correct=%s tools=%d" % (
            time.strftime("%H:%M:%S"), t["q"][:52], r["parsed"], t["expect"], r["correct"], r["tool_calls"])
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    solved = sum(1 for r in results if r["correct"])
    metrics = {"model": MODEL, "tasks": len(TASKS), "solved": solved,
               "accuracy": round(solved / len(TASKS), 4),
               "total_tool_calls": sum(r["tool_calls"] for r in results),
               "per_task": [{k: v for k, v in r.items() if k != "trace"} for r in results]}
    with open(os.path.join(HERE, "results", "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(HERE, "results", "trace.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_task"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
