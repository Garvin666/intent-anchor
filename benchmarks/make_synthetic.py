#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_synthetic — 造一份**信号比例已知**的合成语料，用来验证评测器有判别力。

为什么需要它（这是 v1.1 遗留的第 3 个缺口）：
  v1.1 报了「157 例、精确率 3%」这类数字，但语料是私有工作史，别人无法复现，
  只能选择相信或不信。合成语料解决的是另一个问题：**评测器本身准不准**。
  如果给它一份「一半配对能命中」的语料，它必须报出约 50%；报不出来，
  就说明此前那些 3% 是评测器的产物，不是数据的性质。

怎么造的：
  · 台账有 P+1 行任务 → 相邻配对得 P 例（与 eval_m4.load_cases 同口径）
  · 每例配一条记忆记录，正文里**逐字**包含「本任务」的标题 → 保证能被检索到
    （否则会变沉默，而沉默不进精确率分母，控制变量就废了）
  · 信号例：记录里再逐字包含「下一件事」的标题 → 必然命中
  · 噪声例：记录里写一段与「下一件事」用词完全不相交的诱饵 → 必然误报
  · 因此评测出的精确率应当 ≈ 信号例数 / 总例数

用法：
  python make_synthetic.py --out /tmp/syn05 --pairs 40 --signal 0.5
  python eval_m4.py --ledger /tmp/syn05/ledger.md --config /tmp/syn05/probe.config.json \
                    --probe ../scripts/memory_probe.py
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

# 两组用词**刻意不相交**：任务词表与诱饵词表没有任何共同的 2-gram。
# 不相交是这套控制的关键 —— 一旦共享词，噪声例也会蹭到命中，信号比例就测不准。
TASK_WORDS = [
    "星轨校准", "盐水浸泡", "铜箔蚀刻", "玻璃抛光", "陶土烧制", "麻绳编织",
    "木料阴干", "漆面打磨", "锡焊补漏", "石料开凿", "棉布染色", "皮革鞣制",
    "竹篾弯曲", "树脂灌注", "铝材阳极", "丝线缠绕", "蜡模精修", "珐琅点蓝",
    "银丝掐花", "骨料筛分", "砂纸顺纹", "桐油封固", "藤条蒸软", "釉料配比",
    "生铁退火", "箔片压延", "绒线簇绒", "纸浆抄造", "云母剥离", "贝壳镶嵌",
    "羊毛缩绒", "砑光上光", "桦皮缝合", "角料镂刻", "麻胶压合", "陶轮拉坯",
    "锡器锤揲", "蓝染扎结", "竹青刮削", "漆灰裱布", "石砚雕琢", "牙板修整",
]
DECOY_WORDS = [
    "晨间例会", "周报归档", "报销单据", "座位调整", "门禁登记", "快递领取",
    "会议纪要", "值班排班", "名片印制", "工牌补办", "车位续费", "体检预约",
    "年假申请", "差旅预订", "发票核验", "合同盖章", "用章备案", "档案借阅",
    "培训报名", "问卷回收", "茶水补给", "绿植浇水", "白板擦除", "窗帘更换",
    "空调报修", "钥匙交接", "打印耗材", "网络报障", "邮件分流", "访客接待",
    "奖品采购", "横幅制作", "场地勘察", "音响调试", "摄影跟拍", "茶歇结算",
    "桌牌摆放", "签到核对", "合影站位", "撤场清点", "物料入库", "库房盘点",
]


def bigrams(s: str):
    """只取 CJK 2-gram —— 与 eval_m4.grams 同口径。编号（T000）不进这个集合。"""
    out = set()
    for run in re.findall(r"[\u4e00-\u9fff]+", s):
        for i in range(len(run) - 1):
            out.add(run[i:i + 2])
    return out


def verify_vocab():
    """词表自检：三组之间两两不得共享任何 2-gram。

    这是整套控制的命门。第一版忘了这条，所有标题都带「复核」后缀，
    于是**噪声例**也靠这一个 2-gram 拿到 0.2 > 阈值 0.15，全被判成命中 ——
    控制组直接假绿成 100%。词表一污染，后面所有数字都不作数。
    """
    errs = []
    groups = [("任务", TASK_WORDS), ("诱饵", DECOY_WORDS)]
    for name, words in groups:
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                shared = bigrams(words[i]) & bigrams(words[j])
                if shared:
                    errs.append(f"{name}词表内 「{words[i]}」与「{words[j]}」共享 {sorted(shared)}")
    for a in TASK_WORDS:
        for b in DECOY_WORDS:
            shared = bigrams(a) & bigrams(b)
            if shared:
                errs.append(f"任务「{a}」与诱饵「{b}」共享 {sorted(shared)}")
    if errs:
        raise SystemExit("F1 词表未做到两两不相交：\n  " + "\n  ".join(errs))


def build(out: Path, pairs: int, signal: float, seed: int):
    verify_vocab()
    if pairs + 1 > len(TASK_WORDS):
        print(f"F2 pairs 最多 {len(TASK_WORDS) - 1}（词表长度限制）", file=sys.stderr)
        return 2
    rng = random.Random(seed)

    # 标题 = 编号 + 独占词。**不加任何公共后缀** —— 公共后缀就是污染源（见 verify_vocab）。
    titles = [f"T{i:03d}{TASK_WORDS[i]}" for i in range(pairs + 1)]

    k = round(signal * pairs)                      # 信号例数：取整，保证期望值可算
    sig_idx = set(rng.sample(range(pairs), k))     # 哪几例是信号

    # ---- 台账：P+1 行，eval_m4 会配成 P 例
    rows = ["| 日期 | 任务 | 状态 |", "| --- | --- | --- |"]
    for i, t in enumerate(titles):
        rows.append(f"| 2026-01-{(i % 28) + 1:02d} | {t} | 完成 |")
    (out / "ledger.md").write_text("\n".join(rows) + "\n", encoding="utf-8", newline="")

    # ---- 记忆：每例一条记录，落在 handoffs 层
    rec_dir = out / "memory" / "handoffs" / "syn"
    rec_dir.mkdir(parents=True, exist_ok=True)
    (out / "memory" / "profile").mkdir(parents=True, exist_ok=True)
    (out / "memory" / "profile" / "working-style.md").write_text(
        "# 合成语料 · 工作方式\n\n本文件仅用于占位，让常驻层路径存在。\n",
        encoding="utf-8", newline="")

    for i in range(pairs):
        # ⚠️ 第二处命门：**下一件事只写任务词、不带编号**。
        # 若写成完整标题 titles[i+1]，那么 rec(i+1) 与 rec(i) 会同时包含它，
        # 查 titles[i] 时 rec(i-1) 与 rec(i) 打平，按文件名排序取到 rec(i-1) ——
        # 于是「信号例」被算成误报，实测精确率 30% 而不是 50%（第一版就是这样）。
        # 让当前任务带编号、下一件事不带，top-1 才唯一。
        if i in sig_idx:
            nxt = TASK_WORDS[i + 1]
        else:
            # 诱饵：与下一件事用词完全不相交，故必然判为误报
            nxt = DECOY_WORDS[i % len(DECOY_WORDS)]
        tail = f"未决：{titles[i]} 之后还要接着处理 {nxt} 这一项。"
        # ⚠️ 标题**只出现在「后续」这一行**，别处一律不写，保证 top-1 就是这一行。
        body = (
            f"# 交接 · 合成 · 第 {i:03d} 例\n\n"
            f"## 本次工作的内容\n\n"
            f"主体部分已完成，验证方式为脚本自检，产物已落盘。\n\n"
            f"## 结果\n\n"
            f"自检通过，无遗留问题。\n\n"
            f"## 后续事项\n\n"
            f"{tail}\n"
        )
        (rec_dir / f"rec-{i:03d}.md").write_text(body, encoding="utf-8", newline="")

    cfg = {
        "_comment": "由 make_synthetic.py 生成，供 eval_m4 使用",
        "kb_root": str(out),
        "layers": [
            {"name": "交接", "weight": 1.0, "glob": "memory/handoffs/**/*.md"},
            {"name": "画像", "weight": 1.5, "glob": "memory/profile/*.md"},
        ],
        "resident": {"files": ["memory/profile/working-style.md"], "ledger": ""},
        "semantic": {"enabled": False},
    }
    (out / "probe.config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")

    expect = k / pairs
    meta = {
        "pairs": pairs,
        "signal_ratio_setting": signal,
        "signal_cases": k,
        "expected_precision": round(expect, 4),
        "seed": seed,
        "note": "评测出的精确率应 ≈ expected_precision；发声率应 = 100%（否则控制变量被污染）",
    }
    (out / "expected.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")
    print(f"[OK] 合成语料 → {out}")
    print(f"     台账 {pairs + 1} 行 → {pairs} 例；信号 {k} 例（设定 {signal}）")
    print(f"     期望精确率 {expect:.0%}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=40)
    ap.add_argument("--signal", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=20261104)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    return build(out, a.pairs, a.signal, a.seed)


if __name__ == "__main__":
    sys.exit(main())
