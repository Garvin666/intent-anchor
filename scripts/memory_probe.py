#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""memory_probe — 给 intent-anchor 用的本地记忆探针（零第三方依赖）

三个子命令：
  pin       输出「常驻层」——每次开工都要读的那一小撮文件
  probe     按查询检索「检索层」——字面 2/3-gram + 可选语义通道
  selftest  自检（阳性 / 阴性 / 行号自洽 / 语义通道）

设计口径（改动前先读）：
  1. **零第三方依赖**，只用标准库。语义通道靠 subprocess 委托外部检索引擎，
     不新增任何 import 级依赖 —— 复用而不是再造一套向量索引。
  2. **只读**，从不写记忆库。
  3. **无命中就说无命中**，不构造兜底结果 —— 给出的一切都必须可溯源到「路径:行号」。
  4. **降级必须出声**，禁止静默降级 —— 静默降级会把「没查」伪装成「查过了」。

配置：默认读脚本同目录的 `memory_probe.config.json`，也可 `--config` 指定或
用环境变量 `INTENT_ANCHOR_CONFIG` 指向。`--root` 优先级最高，用于临时换库。

退出码：0 成功 / 1 自检未通过 / 2 参数或库不可读 / 3 语义通道强制要求但不可用
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------- 配置

FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.S)
ENTRY_HEAD = re.compile(r"^#{2,3}\s*#?\d+\s*·", re.M)
CJK_RUN = re.compile(r"[\u4e00-\u9fff]+")
EN_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_.\-]{1,}")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_NAMES = ("memory_probe.config.json", "memory_probe.config.example.json")


def find_config(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.is_file() else None
    env = os.environ.get("INTENT_ANCHOR_CONFIG")
    if env and Path(env).is_file():
        return Path(env)
    for base in (SCRIPT_DIR, Path.cwd()):
        for name in DEFAULT_CONFIG_NAMES:
            p = base / name
            if p.is_file():
                return p
    return None


def load_config(explicit: str | None) -> dict:
    p = find_config(explicit)
    if p is None:
        return {}
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"F1 配置不可解析（{p}）：{e}", file=sys.stderr)
        return {}
    cfg["_path"] = str(p)
    return cfg


def rel_to(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def keep(p: Path) -> bool:
    n = p.name
    try:
        size = p.stat().st_size
    except OSError:
        return False
    return not (".bak" in n or n.lower() == "readme.md" or size > 4_000_000)


# ---------------------------------------------------------------- 层

DEFAULT_LAYER = {"weight": 1.0}


def layer_specs(cfg: dict) -> list[dict]:
    """配置里的 layers；缺省则把整个库当一层（只保底，不建议）。"""
    specs = cfg.get("layers") or []
    if specs:
        return specs
    return [{"name": "记忆", "weight": 1.0, "glob": "**/*.md"}]


def expand(root: Path, pattern: str) -> list[Path]:
    """支持 `**/` 递归与 `*` 通配；返回去重后的文件列表。"""
    if "**" in pattern:
        base = pattern.split("**")[0].rstrip("/")
        tail = pattern.split("**", 1)[1].lstrip("/")
        start = root / base if base else root
        if not start.is_dir():
            return []
        if not tail or tail == "*.md":
            return sorted(p for p in start.rglob("*.md") if p.is_file())
        return sorted(p for p in start.rglob(tail) if p.is_file())
    if "*" in pattern:
        parent = root / Path(pattern).parent
        name = Path(pattern).name
        if not parent.is_dir():
            return []
        return sorted(p for p in parent.iterdir()
                      if p.is_file() and fnmatch.fnmatch(p.name, name))
    p = root / pattern
    return [p] if p.is_file() else []


def layers(root: Path, cfg: dict) -> dict:
    """返回 {层名: (权重, [文件…])}。缺失的 glob 自动跳过，不报错。"""
    out: dict[str, tuple[float, list[Path]]] = {}
    for spec in layer_specs(cfg):
        name = spec.get("name") or "记忆"
        files = [p for p in expand(root, spec.get("glob", "**/*.md")) if keep(p)]
        if files:
            w = float(spec.get("weight", DEFAULT_LAYER["weight"]))
            if name in out:  # 同名层合并
                out[name] = (out[name][0], out[name][1] + files)
            else:
                out[name] = (w, files)
    return out


# ---------------------------------------------------------------- 读取

def read_text(p: Path) -> str:
    try:
        return FRONTMATTER.sub("", p.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return ""


def read_text_with_offset(p: Path):
    """返回 (剥 frontmatter 后的正文, 被剥掉的行数)。

    行号必须加回 offset 才是文件真实行号 —— 否则「依据：路径:行号」会系统性偏移
    frontmatter 的长度。这是本探针宁可多写两行也不能省的地方：
    只验「路径存在」的检查器抓不到这个错，它照样全绿。
    """
    try:
        raw = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "", 0
    m = FRONTMATTER.match(raw)
    if not m:
        return raw, 0
    return raw[m.end():], m.group(0).count("\n")


def split_entries(text: str) -> list[str]:
    """按 `## #N ·` / `### #N ·` 切分条目标题块。"""
    idx = [m.start() for m in ENTRY_HEAD.finditer(text)]
    if not idx:
        return [text]
    out = []
    if idx[0] > 0:
        out.append(text[:idx[0]].rstrip())
    for i, s in enumerate(idx):
        e = idx[i + 1] if i + 1 < len(idx) else len(text)
        out.append(text[s:e].rstrip())
    return [b for b in out if b.strip()]


# ---------------------------------------------------------------- 语义通道（委托外部引擎）

def semantic_cfg(cfg: dict) -> dict:
    return cfg.get("semantic") or {}


def semantic_search(query: str, k: int, root: Path, cfg: dict):
    """委托外部检索引擎做语义召回。返回 (data, err, reason)。

    err 非空时 reason 区分两种情况：
      "disabled" —— 用户没配（不是故障）
      "broken"   —— 配了但跑不起来（是真故障，必须报出来）
    """
    sc = semantic_cfg(cfg)
    if not sc.get("enabled"):
        return None, "语义通道未配置（config.semantic.enabled 未开启）", "disabled"

    py = sc.get("python")
    home = sc.get("module_home")
    module = sc.get("module") or ""
    zone = sc.get("zone")
    timeout = int(sc.get("timeout", 30))

    if not module:
        return None, "配置缺 semantic.module（检索引擎的模块名）", "broken"
    if not py or not Path(py).is_file():
        return None, f"缺解释器 {py}", "broken"
    if not home or not Path(home).is_dir():
        return None, f"缺检索引擎源码目录 {home}", "broken"

    env = dict(os.environ)
    env["PYTHONPATH"] = str(home)
    # --root 是引擎的顶层参数，必须放在子命令**之前**
    cmd = [str(py), "-m", module, "--root", str(root), "search", query, "-k", str(max(k, 8)),
           "--json", "--no-log"]
    if zone:
        cmd += ["--zone", zone]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return None, f"超时（>{timeout}s，模型可能正在冷加载）", "broken"
    except OSError as e:
        return None, f"启动失败：{e}", "broken"

    if r.returncode != 0:
        lines = r.stderr.decode("utf-8", "replace").strip().splitlines()
        return None, f"退出码 {r.returncode}：{lines[-1] if lines else '无输出'}", "broken"
    try:
        return json.loads(r.stdout.decode("utf-8", "replace")), None, ""
    except json.JSONDecodeError as e:
        return None, f"输出非 JSON：{e}", "broken"


# ---------------------------------------------------------------- 行号回贴

ANCHOR_MIN = 14


def locate_line(p: Path, snippet: str):
    """把 chunk 级 snippet 落回文件真实行号；定位不到返回 None（不猜）。

    这一步是语义通道能用的前提：引擎只给 rel_path + chunk，而本技能的硬契约是
    「依据＝路径:行号」。没有回贴，整个可溯源机制就退化。

    ⚠️ 短片段不能当锚点：「2026-10-12」这类会命中整表任意一行（实测 32 条里 12 条贴错）。
    故锚点下限 ANCHOR_MIN，且长片段优先、长度逐级降。
    """
    try:
        raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None

    core = (snippet or "").replace("…", " ")
    cells = [x.strip() for x in re.split(r"[|｜•]", core)]
    cells = sorted([x for x in cells if len(x) >= ANCHOR_MIN], key=len, reverse=True)
    if not cells:
        whole = core.strip()
        cells = [whole] if len(whole) >= ANCHOR_MIN else []

    for cell in cells:
        for n in (32, 24, 18, ANCHOR_MIN):
            needle = cell[:n]
            if len(needle) < ANCHOR_MIN:
                continue
            for i, line in enumerate(raw, 1):
                if needle in line:
                    return i
    return None


def locate_heading(p: Path, heading: str):
    """snippet 定位失败时的兜底：按 chunk 的小节标题定位到标题行。"""
    if not heading:
        return None
    leaf = str(heading).split(">")[-1].strip()
    if len(leaf) < 4:
        return None
    try:
        raw = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for i, line in enumerate(raw, 1):
        if leaf in line and line.lstrip().startswith("#"):
            return i
    return None


# ---------------------------------------------------------------- 分词与打分

def tokenize(q: str, cap: int = 48):
    """英文按词、中文按 2-gram 与 3-gram 滑窗。返回 {token: 权重}。"""
    toks: dict[str, int] = {}
    for w in EN_WORD.findall(q):
        w = w.lower()
        if len(w) >= 2:
            toks[w] = max(toks.get(w, 0), min(len(w), 8))
    for run in CJK_RUN.findall(q):
        n = len(run)
        if n <= 3:
            if n >= 2:
                toks[run] = max(toks.get(run, 0), n)
            continue
        for i in range(n - 1):
            toks[run[i:i + 2]] = max(toks.get(run[i:i + 2], 0), 2)
        for i in range(n - 2):
            toks[run[i:i + 3]] = max(toks.get(run[i:i + 3], 0), 3)
    if len(toks) > cap:
        ranked = sorted(toks.items(), key=lambda kv: (-kv[1], kv[0]))[:cap]
        toks = dict(ranked)
    return toks


def score_line(line: str, toks: dict, stem_low: str) -> float:
    low = line.lower()
    hit = 0
    score = 0.0
    for t, w in toks.items():
        if t in low:
            hit += 1
            score += w
    if hit == 0:
        return 0.0
    # 文件名本身是语义标签，命中文件名说明整篇相关，给额外加成
    for t, w in toks.items():
        if t in stem_low:
            score += w * 0.5
            break
    return score


# ---------------------------------------------------------------- pin

def cmd_pin(args, cfg):
    root = Path(args.root)
    resident = cfg.get("resident") or {}
    total = 0
    blocks = []

    def emit(title: str, rel: str, body: str):
        nonlocal total
        lines = body.rstrip("\n").splitlines()
        if not lines:
            return
        total += len(lines)
        blocks.append((title, rel, lines))

    for item in resident.get("files") or []:
        p = root / item
        if p.is_file():
            emit(f"{p.stem}（常驻：每次开工必读）", rel_to(p, root), read_text(p))

    for pattern in resident.get("globs") or []:
        for p in expand(root, pattern):
            if keep(p):
                emit(f"{p.stem}（常驻）", rel_to(p, root), read_text(p))

    ledger_rel = resident.get("ledger")
    if ledger_rel:
        p = root / ledger_rel
        if p.is_file():
            tail = split_entries(read_text(p))[-max(1, args.tail_entries):]
            emit(f"{p.stem} · 最近 {len(tail)} 条", rel_to(p, root), "\n\n".join(tail))

    print(f"# 记忆常驻层（预算 {args.budget} 行）\n")
    if not blocks:
        print("（配置里的 resident 项都不存在 —— 检查 memory_probe.config.json）\n")
    for title, rel, lines in blocks:
        print(f"## {title}\n来源: {rel}（{len(lines)} 行）\n")
        print("\n".join(lines))
        print()

    print("## 可检索层规模（用 probe --query 按需取，不要整读）")
    for spec in layer_specs(cfg):
        n = len([p for p in expand(root, spec.get("glob", "")) if keep(p)])
        print(f"- {spec.get('name', '记忆')}: {n} 篇")
    print()
    print(f"常驻层合计: {total} 行 / 预算 {args.budget} 行 → "
          f"{'在预算内' if total <= args.budget else '超出预算，需裁剪 --tail-entries 或常驻文件'}")
    return 0 if total <= args.budget else 3


# ---------------------------------------------------------------- probe

def cmd_probe(args, cfg):
    root = Path(args.root)
    all_layers = layers(root, cfg)
    if not all_layers:
        print("F0 记忆库不可读：配置里的 glob 一个文件都没匹配到，请检查 --root 与 config.layers",
              file=sys.stderr)
        return 2

    wanted = None
    if args.kind:
        wanted = {k.strip() for k in args.kind.split(",") if k.strip()}
        unknown = wanted - set(all_layers)
        if unknown:
            print(f"未知层名: {sorted(unknown)}；可选: {sorted(all_layers)}", file=sys.stderr)
            return 2

    toks = tokenize(args.query)
    if not toks:
        print("查询无有效词元（至少 2 个汉字或 2 个字母）", file=sys.stderr)
        return 2

    hits = []
    for name, (weight, files) in all_layers.items():
        if wanted and name not in wanted:
            continue
        for p in files:
            text, off = read_text_with_offset(p)
            if not text:
                continue
            stem_low = p.stem.lower()
            best: dict[int, list] = {}
            for ln, line in enumerate(text.splitlines(), 1):
                if len(line) > 400 or not line.strip():
                    continue
                s = score_line(line, toks, stem_low)
                if s >= args.min_score:
                    best.setdefault(round(s), []).append((ln + off, line.strip()))
            for s in sorted(best, reverse=True):
                for ln, line in best[s]:
                    hits.append({"layer": name, "path": rel_to(p, root), "line": ln,
                                 "score": round(s * weight, 2), "text": line[:args.excerpt]})

    hits.sort(key=lambda h: (-h["score"], h["path"], h["line"]))

    # 相对阈值：短指令下「修复」「这个」这类常见词会带出大量单 token 噪音。
    # 只保留与最高分同一量级的命中，否则一条 18 分与一堆 3 分并列，等于没排序。
    cutoff = args.min_score
    if hits:
        cutoff = max(cutoff, round(hits[0]["score"] * args.best_ratio, 2))
        hits = [h for h in hits if h["score"] >= cutoff]

    # 同一文件最多取 2 条，避免一篇长文淹没结果
    per_file: dict[str, int] = {}
    picked = []
    for h in hits:
        c = per_file.get(h["path"], 0)
        if c >= 2:
            continue
        per_file[h["path"]] = c + 1
        picked.append(h)
        if len(picked) >= args.top:
            break
    for h in picked:
        h["src"] = "字面"

    # ---------------------------------------------------------------- 语义通道
    sem_hits: list[dict] = []
    sem_err = sem_reason = None
    sem_meta: dict = {}
    if args.semantic != "off":
        data, sem_err, sem_reason = semantic_search(args.query, args.top, root, cfg)
        if sem_err:
            if args.semantic == "on":
                print(f"F3 语义通道不可用（--semantic on 要求必须可用）：{sem_err}", file=sys.stderr)
                return 3
        else:
            basis = data.get("no_answer_basis") or {}
            sem_meta = {
                "no_answer": data.get("no_answer"),
                "top1_cosine": basis.get("value"),
                "threshold": basis.get("threshold"),
                "mode": data.get("mode"),
                "vector_used": data.get("vector_used"),
                "returned": data.get("returned"),
            }
            lex_keys = {(h["path"], h["line"]) for h in picked}
            for h in data.get("hits", []):
                rel = h.get("rel_path") or ""
                sp = root / rel
                if not rel or not sp.is_file():
                    continue
                ln = locate_line(sp, h.get("snippet", ""))
                if ln is None:
                    ln = locate_heading(sp, h.get("heading", ""))
                if ln is None or (rel, ln) in lex_keys:
                    continue
                sem_hits.append({
                    "layer": "语义", "path": rel, "line": ln,
                    "score": round(float(h.get("score") or 0), 4),
                    "text": (h.get("snippet") or "").replace("\n", " ")[:args.excerpt],
                    "src": "语义", "heading": h.get("heading", ""),
                })
                if len(sem_hits) >= args.top:
                    break

    scanned = sum(len(f) for _, f in all_layers.values())

    # ---------------------------------------------------------------- 无命中
    if not picked and not sem_hits:
        if sem_meta.get("no_answer") is True:
            verdict = (f"语义层判无相关（top1 余弦 {sem_meta.get('top1_cosine')} "
                       f"< 阈值 {sem_meta.get('threshold')}）")
        elif sem_err:
            verdict = f"语义通道{('未配置' if sem_reason == 'disabled' else '不可用')}（{sem_err}）—— 本次只有字面判定，结论强度较弱"
        elif sem_meta:
            verdict = (f"语义层有召回但行号定位全部失败（returned={sem_meta.get('returned')}）"
                       "—— 不可用作依据，请人工翻该文件")
        else:
            verdict = "未启用语义通道，仅有字面判定"

        msg = (f"记忆库无相关条目（层: {sorted(wanted) if wanted else '全部'}，"
               f"词元 {len(toks)} 个，有效阈值 {cutoff}）；{verdict}")
        if args.json:
            print(json.dumps({"query": args.query, "hits": [], "semantic_hits": [],
                              "semantic_meta": sem_meta or {"error": sem_err},
                              "note": msg}, ensure_ascii=False, indent=2))
        else:
            print(msg)
            print("→ 结论：此项属真空白，可以提问。不得编造记忆依据。")
        return 0

    if args.json:
        print(json.dumps({"query": args.query, "hits": picked, "semantic_hits": sem_hits,
                          "semantic_meta": sem_meta or ({"error": sem_err} if sem_err else {}),
                          "scanned": scanned, "cutoff": cutoff},
                         ensure_ascii=False, indent=2))
        return 0

    print(f"# 记忆命中 · query={args.query!r} · 扫描 {scanned} 个文件\n")
    if picked:
        print(f"## 字面通道（{len(picked)} 条，有效阈值 {cutoff}）")
        for h in picked:
            print(f"[{h['score']:>6}] {h['layer']} | {h['path']}:{h['line']}")
            print(f"         {h['text']}")
    else:
        print("## 字面通道（0 条）")

    print()
    if sem_err:
        tag = "未配置" if sem_reason == "disabled" else "不可用"
        print(f"## 语义通道 · **{tag}**（{sem_err}）")
        print("         ⚠️ 已退回字面通道 —— 结论强度较弱，引用时须说明「未做语义核对」。")
    else:
        print(f"## 语义通道 · 委托外部引擎（{len(sem_hits)} 条，top1 余弦 "
              f"{sem_meta.get('top1_cosine')} / 拒答阈值 {sem_meta.get('threshold')}，"
              f"no_answer={sem_meta.get('no_answer')}）")
        for h in sem_hits:
            print(f"[{h['score']:>6}] 语义 | {h['path']}:{h['line']}")
            print(f"         {h['text']}")
        if not sem_hits:
            print("         （语义命中与字面重合，或行号定位失败）")

    print("\n用法提醒：引用时写 `[记忆] 路径:行号`；上面没有的，一律标 `[推断]`。")
    return 0


# ---------------------------------------------------------------- selftest

def _run_probe(root: Path, cfg: dict, q: str, semantic: str = "off", excerpt: int = 100):
    import io
    from contextlib import redirect_stdout
    argv = argparse.Namespace(root=str(root), query=q, top=15, min_score=2.5,
                              best_ratio=0.35, kind=None, excerpt=excerpt, json=True,
                              semantic=semantic)
    buf = io.StringIO()
    with redirect_stdout(buf):
        cmd_probe(argv, cfg)
    return json.loads(buf.getvalue() or "{}")


def _self_queries(root: Path, cfg: dict, n: int = 3):
    """从语料自身取查询：任取一个文件里的一行，截其中一段当查询，必然可命中自己。

    为什么不用写死的查询串：写死的查询只对**某一个库**成立，换库就全红，
    而那不是「技能坏了」，是「测试写错了库」。自指查询对任何语料都成立。
    """
    files = [p for _, fs in layers(root, cfg).values() for p in fs]
    files = [p for p in files if keep(p)]
    out = []
    step = max(1, len(files) // max(1, n))
    for p in files[::step]:
        text = read_text(p)
        for line in text.splitlines():
            s = line.strip()
            core = "".join(CJK_RUN.findall(s))
            if len(core) >= 16:
                out.append((core[:12], rel_to(p, root)))
                break
        if len(out) >= n:
            break
    return out


def cmd_selftest(args, cfg):
    root = Path(args.root)
    if not layers(root, cfg):
        print("FAIL 记忆库不可读", file=sys.stderr)
        return 2

    failures = 0
    all_hits = []

    print("== 阳性对照（自指查询，应有命中）==")
    positives = _self_queries(root, cfg)
    if not positives:
        print("FAIL  语料里找不到长度 ≥16 汉字的行，无法构造自指查询")
        failures += 1
    for q, src in positives:
        hits = _run_probe(root, cfg, q).get("hits", [])
        ok = len(hits) >= 1
        print(f"{'PASS' if ok else 'FAIL'}  {q!r} -> {len(hits)} 命中  （取自 {src}）")
        failures += 0 if ok else 1
        all_hits += hits

    print("\n== 阴性对照（应零命中）==")
    negatives = ["zzqxvbnm", "kkjjhhggff", "魑魅魍魉"]
    for q in negatives:
        hits = _run_probe(root, cfg, q).get("hits", [])
        ok = len(hits) == 0
        print(f"{'PASS' if ok else 'FAIL'}  {q!r} -> {len(hits)} 命中（应为 0）")
        failures += 0 if ok else 1

    # 关键：只验「有命中」不验「行号对得上」，等于没验依据可溯源 ——
    # 本探针曾因剥 frontmatter 而让全部行号系统性偏移，路径检查照样全绿。
    print("\n== 行号自洽（命中行必须落在报告的那一行）==")
    bad = 0
    for h in all_hits:
        target = root / h["path"]
        try:
            real = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            print(f"FAIL  文件不可读 {h['path']}")
            bad += 1
            continue
        n = h["line"]
        if n < 1 or n > len(real) or h["text"] not in real[n - 1]:
            print(f"FAIL  {h['path']}:{n} 与该行内容不符")
            bad += 1
    checked = len(all_hits)
    print(f"{'PASS' if bad == 0 else 'FAIL'}  {checked - bad}/{checked} 条命中的行号与内容一致")
    failures += 0 if bad == 0 else 1

    # 语义通道。**不能假设「字面零命中」** —— 口语查询仍会撞上常见 2-gram。
    # 真实价值在「新增召回 + 行号回贴保真」。
    print("\n== 语义通道（可用性 / 新增召回 / 行号回贴保真）==")
    sem_status = "SKIP"
    syn = args.synonym or ["弄一下", "跑不起来", "没法打开"]
    all_sem: list[dict] = []
    usable = None
    for q in syn:
        data = _run_probe(root, cfg, q, semantic="auto", excerpt=600)
        meta = data.get("semantic_meta") or {}
        if meta.get("error"):
            print(f"SKIP  {q!r} -> 语义通道未启用/不可用（{meta.get('error')}）")
            sem_status = "SKIP"
            break
        sem_status = "RUN"
        if usable is None:
            usable = isinstance(meta.get("no_answer"), bool) and meta.get("top1_cosine") is not None
        sem = data.get("semantic_hits", [])
        ok = len(sem) >= 1
        print(f"{'PASS' if ok else 'FAIL'}  {q!r} -> 新增语义召回 {len(sem)} 条"
              f"（余弦 {meta.get('top1_cosine')} / 阈值 {meta.get('threshold')} / "
              f"no_answer={meta.get('no_answer')}）")
        failures += 0 if ok else 1
        all_sem += sem

    if sem_status == "RUN":
        ok = bool(usable)
        print(f"{'PASS' if ok else 'FAIL'}  可用性 -> no_answer 布尔 + top1_cosine 均返回")
        failures += 0 if ok else 1

        # 反向包含：定位到的真实行，回查是否真在引擎给的 chunk 快照里。
        # 判据取「整行任一 12 字窗口出现在快照中」而非「行首 12 字」——
        # 快照是从行中间截的，行首常落在窗口外（初版就是这么误判成 BAD 的）。
        good = badn = 0
        for h in all_sem:
            try:
                real = (root / h["path"]).read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                badn += 1
                continue
            n = h["line"]
            if n < 1 or n > len(real):
                badn += 1
                continue
            line_n = re.sub(r"[`*|#>\-\s]", "", real[n - 1])
            snip = re.sub(r"[`*|#>\-\s]", "", h["text"])
            if not line_n:
                badn += 1
                continue
            if len(line_n) < 12:
                hit = line_n in snip
            else:
                hit = any(line_n[i:i + 12] in snip for i in range(0, len(line_n) - 11, 4))
            if hit:
                good += 1
            else:
                badn += 1
                print(f"       BAD  {h['path']}:{n} 定位行内容不在 chunk 快照内")
        total_sem = good + badn
        ratio = (good / total_sem) if total_sem else 0
        ok = total_sem > 0 and ratio >= 0.9
        print(f"{'PASS' if ok else 'FAIL'}  行号回贴保真 -> {good}/{total_sem} 条定位行可在 chunk 快照中反查到")
        failures += 0 if ok else 1

    if sem_status == "SKIP":
        print("SKIP  语义通道未检测 —— 未检测不等于通过")

    total = (len(positives) + len(negatives) + 1
             + (len(syn) + 2 if sem_status == "RUN" else 0))
    print(f"\n结果: {total - failures}/{total} 通过"
          + ("" if sem_status == "RUN" else f"（语义段 SKIP，分母不含 {len(syn) + 2} 条）"))
    return 0 if failures == 0 else 1


# ---------------------------------------------------------------- 入口

def main(argv=None):
    ap = argparse.ArgumentParser(description="intent-anchor 记忆探针（零依赖）")
    ap.add_argument("--root", default=None, help="知识库根目录（覆盖 config.kb_root）")
    ap.add_argument("--config", default=None, help="配置文件路径")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pin", help="输出常驻层（每次必读）")
    p.add_argument("--budget", type=int, default=700)
    p.add_argument("--tail-entries", type=int, default=3)
    p.set_defaults(func=cmd_pin)

    p = sub.add_parser("probe", help="记忆检索（字面 + 语义）")
    p.add_argument("--query", required=True)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--min-score", type=float, default=2.5)
    p.add_argument("--best-ratio", type=float, default=0.35,
                   help="只保留 ≥ 最高分×该比例的命中（压短指令噪音）")
    p.add_argument("--kind", default=None, help="逗号分隔的层名")
    p.add_argument("--excerpt", type=int, default=100)
    p.add_argument("--semantic", choices=["auto", "on", "off"], default="auto",
                   help="语义通道：auto=不可用则显式降级；on=不可用即失败退出 3；off=只用字面")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("selftest", help="阳性 / 阴性 / 行号自洽 / 语义通道自检")
    p.add_argument("--synonym", action="append",
                   help="语义段用的查询，可重复；缺省用内置口诀式短句")
    p.set_defaults(func=cmd_selftest)

    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    if args.root:
        root = Path(args.root)
    elif cfg.get("kb_root"):
        root = Path(cfg["kb_root"]).expanduser()
        if not root.is_absolute():
            root = Path(cfg["_path"]).resolve().parent / root
    else:
        root = SCRIPT_DIR
    args.root = str(root)

    if not root.is_dir():
        print(f"F0 知识库根不存在：{root}（用 --root 或 config.kb_root 指定）", file=sys.stderr)
        return 2

    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
