from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

# ---------- 日志 ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        # 将日志写入文件
        logging.FileHandler("select_results.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("select")


# ---------- 工具 ----------

def load_data(data_dir: Path, codes: Iterable[str]) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for code in codes:
        fp = data_dir / f"{code}.csv"
        if not fp.exists():
            logger.warning("%s 不存在，跳过", fp.name)
            continue
        df = pd.read_csv(fp, parse_dates=["date"]).sort_values("date")
        # 检查数据是否为空或只有标题行
        if df.empty or df["date"].isna().all():
            logger.warning("%s 数据为空，跳过", fp.name)
            continue
        frames[code] = df
    return frames


def load_stock_info(stocklist_path: Path) -> Dict[str, Dict[str, str]]:
    """加载股票信息（代码、名称、地区、行业）"""
    stock_info: Dict[str, Dict[str, str]] = {}
    if not stocklist_path.exists():
        logger.warning("股票列表文件 %s 不存在，无法加载股票信息", stocklist_path)
        return stock_info
    
    try:
        df = pd.read_csv(stocklist_path, dtype=str)
        # 尝试不同的列名
        code_col = None
        name_col = None
        area_col = None
        industry_col = None
        
        for col in ["symbol", "ts_code", "code"]:
            if col in df.columns:
                code_col = col
                break
        
        for col in ["name", "名称"]:
            if col in df.columns:
                name_col = col
                break
        
        for col in ["area", "地区", "地域"]:
            if col in df.columns:
                area_col = col
                break
        
        for col in ["industry", "行业", "行业分类"]:
            if col in df.columns:
                industry_col = col
                break
        
        if code_col and name_col:
            # 构建需要的列列表
            columns = [code_col, name_col]
            if area_col:
                columns.append(area_col)
            if industry_col:
                columns.append(industry_col)
            
            df = df[columns].dropna(subset=[code_col, name_col])
            df[code_col] = df[code_col].astype(str).str.zfill(6)
            # 只保留6位数字代码
            df = df[df[code_col].str.match(r'^\d{6}$')]
            
            # 构建字典
            for _, row in df.iterrows():
                code = row[code_col]
                info = {"name": row[name_col]}
                if area_col and area_col in row:
                    info["area"] = row[area_col]
                if industry_col and industry_col in row:
                    info["industry"] = row[industry_col]
                stock_info[code] = info
            
            logger.info("从 %s 加载了 %d 只股票的信息", stocklist_path, len(stock_info))
        else:
            logger.warning("无法在 %s 中找到代码和名称列", stocklist_path)
    except Exception as e:
        logger.warning("加载股票列表时出错：%s", e)
    
    return stock_info


def load_config(cfg_path: Path) -> List[Dict[str, Any]]:
    if not cfg_path.exists():
        logger.error("配置文件 %s 不存在", cfg_path)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as f:
        cfg_raw = json.load(f)

    # 兼容三种结构：单对象、对象数组、或带 selectors 键
    if isinstance(cfg_raw, list):
        cfgs = cfg_raw
    elif isinstance(cfg_raw, dict) and "selectors" in cfg_raw:
        cfgs = cfg_raw["selectors"]
    else:
        cfgs = [cfg_raw]

    if not cfgs:
        logger.error("configs.json 未定义任何 Selector")
        sys.exit(1)

    return cfgs


def instantiate_selector(cfg: Dict[str, Any]):
    """动态加载 Selector 类并实例化"""
    cls_name: str = cfg.get("class")
    if not cls_name:
        raise ValueError("缺少 class 字段")

    try:
        module = importlib.import_module("Selector")
        cls = getattr(module, cls_name)
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"无法加载 Selector.{cls_name}: {e}") from e

    params = cfg.get("params", {})
    return cfg.get("alias", cls_name), cls(**params)


# ---------- 主函数 ----------

def main():
    p = argparse.ArgumentParser(description="Run selectors defined in configs.json")
    p.add_argument("--data-dir", default="./data", help="CSV 行情目录")
    p.add_argument("--config", default="./configs.json", help="Selector 配置文件")
    p.add_argument("--stocklist", default="./stocklist.csv", help="股票列表文件（用于获取股票名称）")
    p.add_argument("--date", help="交易日 YYYY-MM-DD；缺省=数据最新日期")
    p.add_argument("--tickers", default="all", help="'all' 或逗号分隔股票代码列表")
    args = p.parse_args()

    # --- 加载股票信息 ---
    stock_info = load_stock_info(Path(args.stocklist))

    # --- 加载行情 ---
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("数据目录 %s 不存在", data_dir)
        sys.exit(1)

    codes = (
        [f.stem for f in data_dir.glob("*.csv")]
        if args.tickers.lower() == "all"
        else [c.strip() for c in args.tickers.split(",") if c.strip()]
    )
    if not codes:
        logger.error("股票池为空！")
        sys.exit(1)

    data = load_data(data_dir, codes)
    if not data:
        logger.error("未能加载任何行情数据")
        sys.exit(1)

    if args.date:
        trade_date = pd.to_datetime(args.date)
    else:
        # 计算所有有效数据的最大日期
        dates = [df["date"].max() for df in data.values() if not df.empty and df["date"].notna().any()]
        if not dates:
            logger.error("所有数据都没有有效的日期")
            sys.exit(1)
        trade_date = max(dates)
        logger.info("未指定 --date，使用最近日期 %s", trade_date.date())

    # --- 加载 Selector 配置 ---
    selector_cfgs = load_config(Path(args.config))

    # --- 逐个 Selector 运行 ---
    for cfg in selector_cfgs:
        if cfg.get("activate", True) is False:
            continue
        try:
            alias, selector = instantiate_selector(cfg)
        except Exception as e:
            logger.error("跳过配置 %s：%s", cfg, e)
            continue

        picks = selector.select(trade_date, data)

        # 将结果写入日志，同时输出到控制台
        logger.info("")
        logger.info("============== 选股结果 [%s] ==============", alias)
        logger.info("交易日: %s", trade_date.date())
        logger.info("符合条件股票数: %d", len(picks))
        
        if picks:
            # 格式化输出：每个股票一行显示
            for idx, code in enumerate(picks, 1):
                info = stock_info.get(code, {})
                name = info.get("name", "")
                area = info.get("area", "")
                industry = info.get("industry", "")
                
                if name or area or industry:
                    parts = [f"{idx}.", code]
                    if name:
                        parts.append(name)
                    if area:
                        parts.append(f"({area})")
                    if industry:
                        parts.append(f"[{industry}]")
                    logger.info("  %s", " ".join(parts))
                else:
                    logger.info("  %d. %s", idx, code)
        else:
            logger.info("无符合条件股票")


if __name__ == "__main__":
    main()
