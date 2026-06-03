#!/usr/bin/env python3
"""
精灵图片爬虫 — 从多个搜索引擎采集精灵参考图片

用法:
    python scripts/crawl_sprites.py                          # 全部8种精灵
    python scripts/crawl_sprites.py --sprites xiao_dujiaoshou  # 仅小独角兽
    python scripts/crawl_sprites.py --limit 100 --proxy http://127.0.0.1:7890
    python scripts/crawl_sprites.py --dry-run                # 预览模式

搜索引擎: Bing (默认) + 百度 (中文内容最佳)
"""

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List
from urllib.parse import quote, urlencode

# Windows GBK 终端兼容
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                   encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer,
                                   encoding="utf-8", errors="replace")

import requests
from PIL import Image

# ============================================================
# HTTP 会话 (支持代理)
# ============================================================

_session = None


def get_session(proxy: str = None) -> requests.Session:
    global _session
    if _session is not None:
        return _session

    _session = requests.Session()
    _session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Accept-Encoding": "gzip, deflate",
    })
    _session.timeout = 15
    if proxy:
        _session.proxies = {"http": proxy, "https": proxy}
        print(f"[proxy] Using: {proxy}")
    return _session


# ============================================================
# 精灵配置
# ============================================================

SPRITE_CONFIG = {
    "huzhu_quan": {
        "name_cn": "护主犬",
        "queries": [
            "洛克王国世界 护主犬 精灵",
            "洛克王国 音速犬 护主犬",
            "洛克王国世界 护主犬 捕捉",
        ],
    },
    "yibei_er": {
        "name_cn": "伊贝儿",
        "queries": [
            "洛克王国世界 伊贝儿 精灵",
            "洛克王国 伊贝儿 精灵",
            "洛克王国世界 伊贝儿 捕捉",
        ],
    },
    "emo_ding": {
        "name_cn": "恶魔叮",
        "queries": [
            "洛克王国世界 恶魔叮 精灵",
            "洛克王国 叮叮恶魔 恶魔叮",
            "洛克王国世界 恶魔叮 捕捉",
        ],
    },
    "juhua_li": {
        "name_cn": "菊花梨",
        "queries": [
            "洛克王国世界 菊花梨 精灵",
            "洛克王国 菊花梨 精灵",
            "洛克王国世界 菊花梨 捕捉",
        ],
    },
    "gongping_ge": {
        "name_cn": "公平鸽",
        "queries": [
            "洛克王国世界 公平鸽 精灵",
            "洛克王国 公平鸽 精灵",
            "洛克王国世界 公平鸽 捕捉",
        ],
    },
    "ling_hu": {
        "name_cn": "灵狐",
        "queries": [
            "洛克王国世界 灵狐 精灵",
            "洛克王国 灵狐 尖嘴狐仙",
            "洛克王国世界 灵狐 捕捉",
        ],
    },
    "xiao_dujiaoshou": {
        "name_cn": "小独角兽",
        "queries": [
            "洛克王国世界 小独角兽 精灵",
            "洛克王国 小独角兽 白金独角兽",
            "洛克王国世界 独角兽 捕捉",
        ],
    },
    "xiaoye_yifu": {
        "name_cn": "小夜",
        "queries": [
            "洛克王国世界 小夜 精灵 朔夜伊芙",
            "洛克王国 朔夜伊芙 小夜",
            "洛克王国世界 小夜 捕捉",
        ],
    },
}


# ============================================================
# 百度图片搜索 (中文内容最佳)
# ============================================================

def search_baidu(query: str, limit: int = 50,
                  proxy: str = None) -> List[str]:
    """百度图片搜索 — 对中文游戏内容效果最好。"""
    sess = get_session(proxy)
    urls = []

    for pn in range(0, min(limit, 120), 30):
        params = {
            "tn": "resultjson_com",
            "ipn": "rj",
            "word": query,
            "pn": pn,
            "rn": min(30, limit - pn),
            "ie": "utf-8",
            "oe": "utf-8",
        }
        url = "https://image.baidu.com/search/acjson?" + urlencode(params)
        try:
            resp = sess.get(url, timeout=15)

            # 百度返回特殊 JSON: objURL 是加密的，middleURL/thumbURL 可直接用
            text = resp.text
            # 优先用 middleURL (百度 CDN 缩略图，可直接下载)
            found = re.findall(r'"middleURL"\s*:\s*"([^"]+)"', text)
            if not found:
                found = re.findall(r'"thumbURL"\s*:\s*"([^"]+)"', text)

            urls.extend(found)
            print(f"    百度 pn={pn}: {len(found)} 张")

            if len(found) < 10:
                break

            time.sleep(1.5)

        except Exception as e:
            print(f"    百度搜索异常: {e}")
            break

    return list(dict.fromkeys(urls))[:limit]


# ============================================================
# Bing 图片搜索
# ============================================================

def search_bing(query: str, limit: int = 50,
                 proxy: str = None) -> List[str]:
    """Bing 图片搜索。"""
    sess = get_session(proxy)
    urls = []

    for offset in range(0, limit, 35):
        params = {
            "q": query,
            "first": offset + 1,
            "count": min(35, limit - offset),
            "qft": "+filterui:photo-photo",
            "form": "IRFLTR",
        }
        try:
            resp = sess.get("https://www.bing.com/images/async",
                            params=params, timeout=20)
            if resp.status_code != 200:
                print(f"    Bing HTTP {resp.status_code}, skipping")
                continue

            # 提取图片 URL
            found = re.findall(r'"murl"\s*:\s*"([^"]+)"', resp.text)
            if not found:
                # 备用: 提取 src 属性中的图片链接
                found = re.findall(
                    r'src="(https?://[^"]+\.(?:jpg|jpeg|png)(?:\?[^"]*)?)"',
                    resp.text, re.IGNORECASE)

            urls.extend(found)
            print(f"    Bing offset={offset}: {len(found)} 张")

            if len(found) < 10:
                break
            time.sleep(1.0)

        except Exception as e:
            print(f"    Bing 搜索异常: {e}")
            break

    return list(dict.fromkeys(urls))[:limit]


# ============================================================
# DuckDuckGo (新版 ddgs 包)
# ============================================================

def search_ddg(query: str, limit: int = 50,
                proxy: str = None) -> List[str]:
    """DuckDuckGo 图片搜索 — 使用新版 ddgs 包。"""
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("    [提示] pip install ddgs 启用 DDG 搜索")
            return []

    urls = []
    try:
        # ddgs 内置代理支持和速率限制
        kwargs = {"max_results": limit}
        if proxy:
            kwargs["proxy"] = proxy

        with DDGS(**kwargs) as ddgs:
            for r in ddgs.images(keywords=query):
                url = r.get("image") or ""
                if url:
                    urls.append(url)

        print(f"    DDG: {len(urls)} 张")

    except Exception as e:
        print(f"    DDG 异常: {e}")

    return urls[:limit]


# ============================================================
# 图片下载
# ============================================================

def download_image(url: str, save_dir: Path, index: int,
                   proxy: str = None,
                   min_size: tuple = (100, 100)) -> bool:
    """下载单张图片并验证。"""
    sess = get_session(proxy)
    try:
        resp = sess.get(url, timeout=15, stream=True)
        if resp.status_code != 200:
            return False

        ct = resp.headers.get("Content-Type", "")
        if "image" not in ct and "octet-stream" not in ct:
            return False

        data = resp.content
        if len(data) < 2048:
            return False

        try:
            img = Image.open(io.BytesIO(data))
            if img.width < min_size[0] or img.height < min_size[1]:
                return False
            if img.width > 8000 or img.height > 8000:
                return False
        except Exception:
            return False

        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        ext = ".jpg"
        if img.format == "PNG":
            ext = ".png"
        elif img.format == "WEBP":
            ext = ".webp"

        filename = f"{index:04d}_{url_hash}{ext}"
        (save_dir / filename).write_bytes(data)
        return True

    except Exception:
        return False


# ============================================================
# 采集主逻辑
# ============================================================

def crawl_sprite(sprite_key: str, limit: int = 50,
                  output_base: str = "dataset",
                  dry_run: bool = False,
                  proxy: str = None) -> int:
    """为指定精灵采集图片。"""
    config = SPRITE_CONFIG.get(sprite_key)
    if not config:
        print(f"[ERROR] 未知精灵: {sprite_key}")
        return 0

    print(f"\n{'=' * 50}")
    print(f"[*] {config['name_cn']} ({sprite_key})")
    print(f"{'=' * 50}")

    save_dir = Path(output_base) / f"raw_{sprite_key}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 多引擎搜索
    all_urls = []
    searchers = [
        ("Baidu", lambda q: search_baidu(q, limit=limit, proxy=proxy)),
        ("Bing", lambda q: search_bing(q, limit=limit, proxy=proxy)),
        ("DDG", lambda q: search_ddg(q, limit=limit, proxy=proxy)),
    ]

    for engine, search_fn in searchers:
        for query in config["queries"]:
            if len(all_urls) >= limit * 3:
                break
            print(f"  [{engine}] {query}")
            urls = search_fn(query)
            all_urls.extend(urls)
            if urls:
                print(f"    -> {len(urls)} URLs (累计 {len(all_urls)})")
        if len(all_urls) >= limit * 2:
            break

    all_urls = list(dict.fromkeys(all_urls))
    print(f"\n  总计: {len(all_urls)} 个唯一图片 URL")

    if dry_run:
        print("  [DRY RUN] 跳过下载，前5个URL:")
        for u in all_urls[:5]:
            print(f"    {u}")
        return len(all_urls)

    # 下载
    downloaded = 0
    for i, url in enumerate(all_urls):
        if downloaded >= limit:
            break
        ok = download_image(url, save_dir, downloaded + 1, proxy=proxy)
        if ok:
            downloaded += 1
            if downloaded % 10 == 0:
                print(f"  下载中: {downloaded}/{limit}")

    print(f"  [OK] {downloaded}/{limit} 张 -> {save_dir}")
    return downloaded


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="精灵图片爬虫 — 自动搜集训练数据"
    )
    parser.add_argument("--sprites", default=None,
                        help="要采集的精灵 (逗号分隔)，默认全部")
    parser.add_argument("--limit", type=int, default=50,
                        help="每只精灵下载上限 (默认 50)")
    parser.add_argument("--output", default="dataset",
                        help="输出目录 (默认 dataset/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅搜索显示URL，不下载")
    parser.add_argument("--proxy", default=None,
                        help="HTTP 代理 (如 http://127.0.0.1:7890)")
    parser.add_argument("--engine", default="all",
                        choices=["baidu", "bing", "ddg", "all"])
    args = parser.parse_args()

    # 确定精灵列表
    if args.sprites:
        sprites = [s.strip() for s in args.sprites.split(",")]
        invalid = [s for s in sprites if s not in SPRITE_CONFIG]
        if invalid:
            print(f"[ERROR] 未知精灵: {invalid}")
            print(f"  可选: {list(SPRITE_CONFIG.keys())}")
            sys.exit(1)
    else:
        sprites = list(SPRITE_CONFIG.keys())

    print("=" * 60)
    print("洛克王国世界 — 精灵图片爬虫")
    print(f"精灵: {len(sprites)} 种 | 每只上限: {args.limit} | 模式: {'预览' if args.dry_run else '下载'}")
    if args.proxy:
        print(f"代理: {args.proxy}")
    print("=" * 60)

    total = 0
    for sprite_key in sprites:
        count = crawl_sprite(
            sprite_key=sprite_key,
            limit=args.limit,
            output_base=args.output,
            dry_run=args.dry_run,
            proxy=args.proxy,
        )
        total += count

    print(f"\n{'=' * 60}")
    print(f"[DONE] {total} 张图片 -> {Path(args.output).resolve()}")
    print(f"下一步:")
    print(f"  1. 手动筛选: 删除 dataset/raw_*/ 中的无关图片")
    print(f"  2. 标注: python scripts/label_tool.py dataset/raw_xiao_dujiaoshou/")
    print(f"  3. 训练: python scripts/incremental_train.py "
          f"--new-data dataset/raw_xiao_dujiaoshou/ --device cuda")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
