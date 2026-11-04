#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[自研工具] verify_refs — 校验文档里「[记忆] 路径:行号」依据是否可解析

用途: 抽取文档里全部 `[记忆] <path>:<line>` 引用，逐条验证文件存在且行号落在文件行数内
用法: python verify_refs.py <文件…> --root <知识库根>
退出码: 0 全通过 / 1 有不可解析的引用 / 2 参数或文件问题

设计口径: **只判「能不能指到」，不判「引用得对不对」** —— 后者无机器 oracle。
这一条必须写在使用者看得见的地方，否则「全绿」会被读成「内容已核对」。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REF = re.compile(r"\[记忆\]\s*([^\s，；、）)「」]+?):(\d+)")


def main(argv=None):
    ap = argparse.ArgumentParser(description="校验 [记忆] 路径:行号 引用可解析性")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--root", default=os.environ.get("INTENT_ANCHOR_ROOT", "."),
                    help="知识库根目录（默认取环境变量 INTENT_ANCHOR_ROOT，再退当前目录）")
    args = ap.parse_args(argv)

    root = Path(args.root)
    total = bad = 0

    for f in args.files:
        p = Path(f)
        if not p.is_file():
            print(f"[F2] 待检文件不存在: {p}", file=sys.stderr)
            return 2
        text = p.read_text(encoding="utf-8", errors="replace")
        # 反引号只是 Markdown 排版，不构成语义；先剥掉再抽引用，避免引号包在代码跨度里被漏检
        refs = REF.findall(text.replace("`", ""))
        print(f"\n=== {p.name} · {len(refs)} 条引用 ===")
        for rel, ln in refs:
            total += 1
            target = (root / rel) if not Path(rel).is_absolute() else Path(rel)
            if not target.is_file():
                print(f"[FAIL] {rel}:{ln} → 文件不存在")
                bad += 1
                continue
            n = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
            if int(ln) > n:
                print(f"[FAIL] {rel}:{ln} → 行号越界（该文件 {n} 行）")
                bad += 1
            else:
                print(f"[ OK ] {rel}:{ln}")

    print(f"\n结果: {total - bad}/{total} 条引用可解析")
    print("注意：本工具只验「路径存在 + 行号不越界」。内容对不对，必须回读那一行。")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
