#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_m4 — 回放评测「下一步预判」到底准不准（2×2 消融）

为什么要这个脚本：
  「提醒你下一步」听起来有用，但它准不准是可以测的，不该靠感觉。做法是拿一份
  按时间顺序记录了「实际做了什么」的台账（ledger），把相邻两行配成
  「刚结束的任务 → 用户实际做的下一件事」，再用记忆库去猜后一件事。

怎么读结果：
  · 命中 = 猜出来的候选与真实下一件事的 CJK 2-gram 重合度 >= 阈值
  · 误报 = 说了话但没命中；**沉默单列**，既不算命中也不算误报
  · ⚠️ **排序键必须是 F1，不是误报率**。误报率的分母只含「说过话」的例数，
    一个从不说话的实现误报率没有定义 —— 单看误报率会被「闭嘴」刷穿。

用法：
  python eval_m4.py --ledger demo/ledger.md --config demo/memory_probe.config.json
  python eval_m4.py ... --shuffle      # 打乱对照：把真实下一步换成别例的
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

CJK = re.compile(r"[\u4e00-\u9fff]+")
CLEAN = re.compile(r"\*\*|`")
MARKERS = ("下一步", "未决", "待办", "待授权", "未修", "未做", "继续", "遗留", "待推送", "尚未")
AUTOMATION = ("定时", "cron", "自动化任务", "每日推送")


def grams(s, n=2):
    out = set()
    for run in CJK.findall(s or ""):
        for i in range(len(run) - n + 1):
            out.add(run[i:i + n])
    return out


def load_cases(ledger: Path, limit: int = 0):
    """从台账里取「任务」列，相邻两行配成 刚结束 → 下一件事。

    列位置按表头定位（不写死 index）—— 台账的列数各人不同，写死会把日期当任务。
    """
    lines = ledger.read_text(encoding="utf-8").splitlines()
    col = None
    rows = []
    for line in lines:
        if not line.startswith("|"):
            continue
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if col is None:
            if "任务" in cells:
                col = cells.index("任务")
            continue
        if set("".join(cells)) <= set("-: "):
            continue
        if len(cells) <= col:
            continue
        v = CLEAN.sub("", cells[col])
        if v:
            rows.append(v)
    if col is None:
        return []
    cases = []
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        # 下一件若是定时任务，那是排程排的不是人选的，不能当 ground truth
        if any(k in b for k in AUTOMATION):
            continue
        cases.append({"id": f"M4-{len(cases) + 1:02d}",
                      "context": a[:220], "next_actual": b[:160]})
    return cases[:limit] if limit else cases


def probe(probe_py: Path, config: str | None, query: str, semantic: str, top: int = 12):
    cmd = [sys.executable, str(probe_py)]
    if config:
        cmd += ["--config", config]
    cmd += ["probe", "--query", query, "--top", str(top), "--semantic", semantic, "--json"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    return d.get("hits", []) + d.get("semantic_hits", [])


def run(cases, args, semantic: str, use_marker: bool):
    hit = fp = silent = 0
    for c in cases:
        hits = probe(Path(args.probe), args.config, c["context"][:80], semantic)
        cands = [h for h in hits if any(m in h["text"] for m in MARKERS)] if use_marker else hits
        if not cands:
            silent += 1
            continue
        g = grams(c["next_actual"])
        sc = len(grams(cands[0]["text"]) & g) / max(1, len(g))
        if sc >= args.threshold:
            hit += 1
        else:
            fp += 1
    n = len(cases)
    spoken = hit + fp
    prec = hit / spoken if spoken else None
    recall = hit / n if n else 0.0
    f1 = (2 * prec * recall / (prec + recall)) if (prec and recall) else 0.0
    return {"hit": hit, "fp": fp, "silent": silent, "n": n, "spoken": spoken,
            "prec": prec, "recall": recall, "f1": f1,
            "fp_rate": (fp / spoken) if spoken else None,
            "speak_rate": spoken / n if n else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--probe", default=str(Path(__file__).resolve().parent.parent / "scripts" / "memory_probe.py"))
    ap.add_argument("--config", default=None)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true", help="打乱对照（判别力检验）")
    args = ap.parse_args()

    cases = load_cases(Path(args.ledger), args.limit)
    if not cases:
        print("F2 台账里没有可用配对（需要至少两行任务、且下一行不是定时任务）", file=sys.stderr)
        return 2
    print(f"样本 {len(cases)} 例\n")

    combos = [("字面 · 有标记闸", "off", True), ("字面 · 无标记闸", "off", False),
              ("语义 · 有标记闸", "auto", True), ("语义 · 无标记闸", "auto", False)]

    print(f"{'组':<18}{'命中':>5}{'误报':>5}{'沉默':>5}{'发言率':>7}{'误报率':>7}{'精确率':>7}{'F1':>7}")
    rows = {}
    for name, sem, mk in combos:
        r = run(cases, args, sem, mk)
        rows[name] = r
        fpr = "—" if r["fp_rate"] is None else f"{r['fp_rate']:.0%}"
        pr = "—" if r["prec"] is None else f"{r['prec']:.0%}"
        print(f"{name:<18}{r['hit']:>5}{r['fp']:>5}{r['silent']:>5}"
              f"{r['speak_rate']:>7.0%}{fpr:>7}{pr:>7}{r['f1']:>7.3f}")

    if args.shuffle:
        print("\n打乱对照（把真实下一步换成别例的，候选逻辑不变）：")
        rng = random.Random(20260101)
        for name, sem, mk in combos:
            sh = []
            for i, c in enumerate(cases):
                j = rng.randrange(len(cases))
                while j == i and len(cases) > 1:
                    j = rng.randrange(len(cases))
                sh.append({**c, "next_actual": cases[j]["next_actual"]})
            rs = run(sh, args, sem, mk)
            real, fake = rows[name]["prec"], rs["prec"]
            verdict = ("样本不足" if real is None or fake is None
                       else "有判别力" if real - fake >= 0.20
                       else "弱判别力" if real > fake else "无判别力")
            print(f"  {name:<18} 真实 {'—' if real is None else f'{real:.0%}'}  "
                  f"打乱 {'—' if fake is None else f'{fake:.0%}'}  → {verdict}")

    best = max(rows, key=lambda k: rows[k]["f1"])
    print(f"\n按 F1 最优：{best}（F1 {rows[best]['f1']:.3f}）")
    print("提醒：若所有组的 F1 都在 0.10 以下，说明「从记忆预判下一件事」这个任务本身不成立，")
    print("      该做的不是继续调参，而是换个口径（见 SKILL.md §4 M4）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
