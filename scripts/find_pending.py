#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""find_pending — 找「刚结束那个任务」的记录里还挂着的未决项，原文照抄 + 行号

为什么不是「预测下一步」：
  实测过（见 benchmarks/eval_m4.py）：从记忆预判下一件事，top-1 精确率 7%，
  即使排序完美上限也只有 20%。记忆记的是「做过什么」，不含「接下来要做什么」。
  所以本工具不生成任何新句子，只做一件事 —— 把记录里**已经写着**的未决项引来。

怎么找：
  1. 按任务名去各层里找**同一个任务**的记录（文件名与任务名 2-gram 覆盖 ≥ 阈值）
  2. 读该记录里标题含「后续 / 未决 / 下一步 / 待办」的小节
  3. 原样输出，附 `路径:行号`

找不到就退出码 1 —— 沉默是正确输出，不是失败。

用法：
  python find_pending.py --task "把导出脚本的编码问题修掉"
  python find_pending.py --ledger ledger.md          # 取台账最后一行当任务名
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]+")
DATE_PREFIX = re.compile(r"^\d{4}[-_]\d{2}[-_]\d{2}([-_]\d{4})?[-_]?")
SECTION = re.compile(r"^#{2,3}.*(后续|未决|下一步|待办|遗留)")
SCRIPT_DIR = Path(__file__).resolve().parent


def grams(s, n=2):
    out = set()
    for run in CJK.findall(s or ""):
        for i in range(len(run) - n + 1):
            out.add(run[i:i + n])
    return out


def load_cfg(explicit):
    p = Path(explicit) if explicit else None
    if p is None:
        for cand in (SCRIPT_DIR.parent / "memory_probe.config.json",
                     SCRIPT_DIR.parent / "memory_probe.config.example.json"):
            if cand.is_file():
                p = cand
                break
    if p is None or not p.is_file():
        return {}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    cfg["_path"] = str(p)
    return cfg


def expand(root: Path, pattern: str):
    if "**" in pattern:
        base, tail = pattern.split("**", 1)
        start = root / base.rstrip("/") if base.rstrip("/") else root
        if not start.is_dir():
            return []
        tail = tail.lstrip("/")
        return sorted(start.rglob(tail or "*.md"))
    if "*" in pattern:
        parent = root / Path(pattern).parent
        return sorted(p for p in parent.iterdir() if p.is_file()) if parent.is_dir() else []
    p = root / pattern
    return [p] if p.is_file() else []


def sections(path: Path):
    """返回 [(起始行号, 小节标题, [正文行…])]。行号从 1 起，含 YAML 前置块。"""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    out = []
    for i, ln in enumerate(lines):
        if not SECTION.match(ln):
            continue
        body = []
        for nxt in lines[i + 1:]:
            if nxt.startswith("## ") or nxt.startswith("### "):
                break
            if nxt.strip():
                body.append(nxt.rstrip())
        if body:
            out.append((i + 1, ln.strip(), body))
    return out


def resolve_root(cfg, explicit):
    if explicit:
        return Path(explicit)
    if cfg.get("kb_root"):
        r = Path(cfg["kb_root"]).expanduser()
        if not r.is_absolute():
            r = Path(cfg["_path"]).resolve().parent / r
        return r
    return SCRIPT_DIR.parent


def task_from_ledger(p: Path) -> str:
    col = None
    last = ""
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if col is None:
            if "任务" in cells:
                col = cells.index("任务")
            continue
        if set("".join(cells)) <= set("-: ") or len(cells) <= col:
            continue
        if cells[col]:
            last = cells[col]
    return last


def main():
    ap = argparse.ArgumentParser(description="找同一任务记录里还挂着的未决项（原文 + 行号）")
    ap.add_argument("--task", default=None, help="刚结束的任务名")
    ap.add_argument("--ledger", default=None, help="从台账最后一行取任务名")
    ap.add_argument("--root", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--match", type=float, default=0.5, help="文件名与任务名的 2-gram 覆盖阈值")
    ap.add_argument("--max", type=int, default=3)
    args = ap.parse_args()

    task = args.task
    if not task and args.ledger:
        task = task_from_ledger(Path(args.ledger))
    if not task:
        print("F2 需要 --task 或 --ledger", file=sys.stderr)
        return 2

    cfg = load_cfg(args.config)
    root = resolve_root(cfg, args.root)
    specs = cfg.get("layers") or [{"name": "记忆", "glob": "**/*.md"}]

    files = []
    for spec in specs:
        files += [p for p in expand(root, spec.get("glob", "**/*.md"))
                  if p.is_file() and ".bak" not in p.name and p.name.lower() != "readme.md"]

    tg = grams(task)
    if not tg:
        print("F2 任务名里没有汉字，2-gram 匹配无从谈起", file=sys.stderr)
        return 2

    scored = []
    for p in files:
        stem = DATE_PREFIX.sub("", p.stem)
        sg = grams(stem)
        if not sg:
            continue
        s = len(tg & sg) / len(sg)
        if s >= args.match:
            scored.append((s, p))
    scored.sort(key=lambda x: -x[0])

    if not scored:
        print(f"（沉默）没有找到与「{task}」对应的记录（阈值 {args.match:.0%}）—— 这是正确输出，不是失败")
        return 1

    emitted = 0
    for score, p in scored:
        secs = sections(p)
        if not secs:
            continue
        print(f"# 同一任务的记录：{p.relative_to(root).as_posix()}（文件名匹配 {score:.0%}）")
        for ln, title, body in secs:
            print(f"\n{title}  ← 该小节起于第 {ln} 行")
            for b in body:
                print(f"  {b}")
            print(f"  [记忆] {p.relative_to(root).as_posix()}:{ln}")
            emitted += 1
            if emitted >= args.max:
                return 0
    if emitted == 0:
        print("（沉默）匹配到了记录，但里面没有「后续 / 未决 / 下一步 / 待办」小节")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
