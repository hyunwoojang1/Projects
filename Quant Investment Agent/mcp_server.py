"""투자 에이전트 MCP 서버.

Claude Code가 이 서버의 도구들을 직접 호출해서 분석을 수행합니다.
Claude가 무엇을, 언제, 왜 호출할지 스스로 결정합니다.

등록 방법:
  프로젝트 루트의 .mcp.json 이 자동으로 이 서버를 등록합니다.

사용 예시 (Claude Code에서):
  "오늘 TQQQ 진입 분석해줘"
  → Claude가 알아서 도구들을 순서대로 호출
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

server = Server("investment-agent")

# ── 데이터 캐시 (서버 세션 내 재사용) ─────────────────────────────────────────
_cache: dict = {}


def _get_data(lookback_years: int = 30) -> dict:
    if "data" not in _cache:
        from modules.data_collector import DataCollector
        dc = DataCollector()
        _cache["data"] = dc.collect(lookback_years=lookback_years)
        _cache["dc"] = dc
    return _cache["data"]


# ── 도구 목록 ──────────────────────────────────────────────────────────────────

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="collect_data",
            description=(
                "시장 데이터를 수집합니다 (yfinance 30년 + FRED 매크로). "
                "다른 분석 도구를 호출하기 전에 먼저 실행하세요."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "lookback_years": {
                        "type": "integer",
                        "description": "수집할 과거 데이터 연수 (기본 30)",
                        "default": 30,
                    }
                },
            },
        ),
        types.Tool(
            name="get_qqq_signal",
            description=(
                "ML/DL 앙상블로 TQQQ 장기 진입 신호를 분석합니다. "
                "XGBoost, LightGBM, LSTM, Transformer 4개 모델의 결과와 "
                "유사 과거 구간(analogs) 및 예상 수익률을 반환합니다."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_regime",
            description=(
                "HMM(Hidden Markov Model)으로 현재 시장 국면을 분류합니다. "
                "강세/약세/전환 국면과 각 국면의 확률을 반환합니다."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_fear_greed",
            description=(
                "공포탐욕지수를 계산합니다 (0=극도공포, 100=극도탐욕). "
                "7개 컴포넌트(모멘텀, 강도, 폭, 풋콜, 변동성, 안전자산, 정크본드)를 반환합니다."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_sector_rotation",
            description=(
                "11개 섹터 ETF의 모멘텀을 분석해 강세/약세 섹터를 식별합니다. "
                "현재 경기 국면(확장기/수축기 등)도 판단합니다."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="get_macro_score",
            description=(
                "금리, 실업률, CPI 등 매크로 지표를 종합해 위험자산 유불리 점수를 계산합니다 (0~100점)."
            ),
            inputSchema={"type": "object", "properties": {}},
        ),
        types.Tool(
            name="generate_report_cards",
            description=(
                "모든 분석 결과를 인스타그램용 카드 이미지 7장으로 생성합니다. "
                "생성된 파일 경로를 반환합니다."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "object",
                        "description": "모든 모듈의 분석 결과를 담은 dict",
                    }
                },
                "required": ["analysis"],
            },
        ),
        types.Tool(
            name="send_email_report",
            description=(
                "분석 결과를 HTML 이메일로 전송합니다. "
                "EMAIL_RECIPIENT 환경변수에 설정된 주소로 발송합니다."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis": {
                        "type": "object",
                        "description": "모든 모듈의 분석 결과",
                    },
                    "claude_opinion": {
                        "type": "string",
                        "description": "Claude의 최종 종합 의견 (자유 텍스트)",
                    },
                },
                "required": ["analysis", "claude_opinion"],
            },
        ),
    ]


# ── 도구 실행 ──────────────────────────────────────────────────────────────────

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    try:
        result = _dispatch(name, arguments)
        return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2, default=str))]
    except Exception as e:
        return [types.TextContent(type="text", text=json.dumps({"error": str(e)}, ensure_ascii=False))]


def _dispatch(name: str, args: dict):
    if name == "collect_data":
        return _tool_collect_data(args.get("lookback_years", 30))
    if name == "get_qqq_signal":
        return _tool_qqq_signal()
    if name == "get_regime":
        return _tool_regime()
    if name == "get_fear_greed":
        return _tool_fear_greed()
    if name == "get_sector_rotation":
        return _tool_sector()
    if name == "get_macro_score":
        return _tool_macro()
    if name == "generate_report_cards":
        return _tool_generate_cards(args.get("analysis", {}))
    if name == "send_email_report":
        return _tool_send_email(args.get("analysis", {}), args.get("claude_opinion", ""))
    return {"error": f"알 수 없는 도구: {name}"}


# ── 각 도구 구현 ───────────────────────────────────────────────────────────────

def _tool_collect_data(lookback_years: int) -> dict:
    data = _get_data(lookback_years)
    market = data["market"]
    fred   = data["fred"]
    factors = data["factors"]
    return {
        "status":         "success",
        "market_rows":    len(market),
        "fred_rows":      len(fred),
        "wrds_rows":      len(factors),
        "market_start":   str(market.index[0].date()) if len(market) > 0 else None,
        "market_end":     str(market.index[-1].date()) if len(market) > 0 else None,
        "tickers":        [c for c in market.columns if not c.endswith("_vol")][:15],
    }


def _tool_qqq_signal() -> dict:
    data = _get_data()
    from modules import qqq_model
    return qqq_model.predict(data["market"], data["fred"], data["factors"])


def _tool_regime() -> dict:
    data = _get_data()
    dc   = _cache["dc"]
    from modules import regime_model
    factor_returns = dc.compute_factor_returns(data["factors"], data["market"])
    return regime_model.predict(data["market"], factor_returns)


def _tool_fear_greed() -> dict:
    data = _get_data()
    from modules import fear_greed
    return fear_greed.compute(data["market"], data["fred"])


def _tool_sector() -> dict:
    data = _get_data()
    from modules import sector_rotation
    return sector_rotation.compute(data["market"])


def _tool_macro() -> dict:
    data = _get_data()
    from modules import macro_score
    return macro_score.compute(data["fred"], data["market"])


def _tool_generate_cards(analysis: dict) -> dict:
    import yaml
    from datetime import datetime
    from report.card_generator import CardGenerator

    cfg_path = ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    date_str  = datetime.now().strftime("%Y-%m-%d")
    output_dir = ROOT / "output"
    gen = CardGenerator(cfg.get("design", {}), output_dir)
    paths = gen.generate_all(analysis, date_str)
    return {
        "status": "success",
        "card_count": len(paths),
        "output_dir": str(output_dir / f"report_{date_str.replace('-','')}"),
        "files": [str(p) for p in paths],
    }


def _tool_send_email(analysis: dict, claude_opinion: str) -> dict:
    from modules.email_sender import send_report
    return send_report(analysis, claude_opinion)


# ── 서버 진입점 ────────────────────────────────────────────────────────────────

async def _main():
    async with mcp.server.stdio.stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    asyncio.run(_main())
