"""Streamlit 편집 UI — 카드 디자인 실시간 편집 및 미리보기.

실행: streamlit run report/editor.py
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

# Ensure project root is on sys.path when run from repo root or report/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

CONFIG_PATH = ROOT / "config.yaml"


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Quant Signal — 카드 편집기",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Load / save config ────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False)
    st.cache_data.clear()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    cfg = load_config()
    design = cfg.get("design", {})
    cards_toggle = cfg.get("cards", {})

    st.title("📊 Quant Signal — 카드 편집기")

    # ── Sidebar: design controls ──────────────────────────────────────────────
    with st.sidebar:
        st.header("디자인 설정")

        st.subheader("테마")
        theme = st.selectbox("테마", ["dark", "light"],
                             index=0 if design.get("theme") == "dark" else 1)
        if theme == "light":
            design["background_color"] = "#FFFFFF"
            design["text_primary"] = "#111111"
            design["card_bg"] = "#F5F5F5"
        else:
            design["background_color"] = "#0D0D0D"
            design["text_primary"] = "#FFFFFF"
            design["card_bg"] = "#1A1A1A"
        design["theme"] = theme

        st.subheader("색상")
        design["accent_color"] = st.color_picker(
            "강조색 (Accent)", design.get("accent_color", "#00FF88"))
        design["secondary_color"] = st.color_picker(
            "보조색 (Secondary)", design.get("secondary_color", "#FF6B35"))
        design["positive_color"] = st.color_picker(
            "상승색", design.get("positive_color", "#00FF88"))
        design["negative_color"] = st.color_picker(
            "하락색", design.get("negative_color", "#FF4444"))
        design["neutral_color"] = st.color_picker(
            "중립색", design.get("neutral_color", "#FFD700"))

        st.subheader("폰트 크기")
        design["font_size_title"] = st.slider(
            "타이틀", 48, 96, int(design.get("font_size_title", 72)))
        design["font_size_subtitle"] = st.slider(
            "서브타이틀", 32, 72, int(design.get("font_size_subtitle", 48)))
        design["font_size_body"] = st.slider(
            "본문", 24, 52, int(design.get("font_size_body", 36)))
        design["font_size_small"] = st.slider(
            "소문자", 18, 40, int(design.get("font_size_small", 28)))

        st.subheader("레이아웃")
        design["logo_text"] = st.text_input(
            "로고 텍스트", design.get("logo_text", "QUANT SIGNAL"))
        design["logo_position"] = st.selectbox(
            "로고 위치", ["top_right", "top_left", "bottom_right", "bottom_left"],
            index=0)
        design["show_disclaimer"] = st.toggle(
            "면책 고지 표시", value=design.get("show_disclaimer", True))

        st.subheader("카드 ON/OFF")
        cards_toggle["card_01_cover"] = st.toggle("카드 1: 표지", value=cards_toggle.get("card_01_cover", True))
        cards_toggle["card_02_timing"] = st.toggle("카드 2: 타이밍 신호", value=cards_toggle.get("card_02_timing", True))
        cards_toggle["card_03_fear_greed"] = st.toggle("카드 3: 공포탐욕지수", value=cards_toggle.get("card_03_fear_greed", True))
        cards_toggle["card_04_regime"] = st.toggle("카드 4: 시장 레짐", value=cards_toggle.get("card_04_regime", True))
        cards_toggle["card_05_sector"] = st.toggle("카드 5: 섹터 로테이션", value=cards_toggle.get("card_05_sector", True))
        cards_toggle["card_06_macro"] = st.toggle("카드 6: 매크로 환경", value=cards_toggle.get("card_06_macro", True))
        cards_toggle["card_07_disclaimer"] = st.toggle("카드 7: 면책 고지", value=cards_toggle.get("card_07_disclaimer", True))

        if st.button("💾 설정 저장", type="primary"):
            cfg["design"] = design
            cfg["cards"] = cards_toggle
            save_config(cfg)
            st.success("설정이 저장되었습니다.")

    # ── Main area: preview & actions ──────────────────────────────────────────
    col1, col2 = st.columns([3, 1])

    with col2:
        st.subheader("실행")
        selected_date = st.date_input("분석 날짜", value=date.today())
        date_str = selected_date.strftime("%Y-%m-%d")

        run_analysis = st.button("🔄 분석 실행", type="primary", use_container_width=True)
        generate_cards = st.button("🎨 카드 생성", use_container_width=True)
        upload_instagram = st.button("📱 인스타그램 업로드", use_container_width=True)

        st.divider()
        st.subheader("로그")
        log_path = ROOT / "logs" / "analysis_log.json"
        if log_path.exists():
            with open(log_path, "r", encoding="utf-8") as f:
                logs = json.load(f)
            recent = logs[-3:] if isinstance(logs, list) else [logs]
            for entry in reversed(recent):
                st.json(entry)

    with col1:
        st.subheader("카드 미리보기")

        output_dir = ROOT / "output"
        today_dir = output_dir / f"report_{date_str.replace('-', '')}"

        if run_analysis:
            with st.spinner("분석 실행 중..."):
                try:
                    analysis = _run_analysis(date_str)
                    st.session_state["analysis"] = analysis
                    st.success("분석 완료!")
                except Exception as e:
                    st.error(f"분석 실패: {e}")

        if generate_cards:
            analysis = st.session_state.get("analysis")
            if analysis is None:
                st.warning("먼저 분석을 실행하세요.")
            else:
                with st.spinner("카드 생성 중..."):
                    try:
                        cfg_reloaded = load_config()
                        paths = _generate(analysis, cfg_reloaded, date_str)
                        st.success(f"{len(paths)}장 생성 완료!")
                    except Exception as e:
                        st.error(f"카드 생성 실패: {e}")

        if upload_instagram:
            st.info("인스타그램 업로드는 upload/instagram.py를 직접 실행하거나 main.py로 수행하세요.")

        # Show generated cards
        card_files = sorted(today_dir.glob("card_*.png")) if today_dir.exists() else []
        if card_files:
            cols = st.columns(min(3, len(card_files)))
            for i, card_path in enumerate(card_files):
                with cols[i % len(cols)]:
                    st.image(str(card_path), use_container_width=True,
                             caption=card_path.stem)
        else:
            st.info("아직 생성된 카드가 없습니다. '분석 실행' 후 '카드 생성'을 클릭하세요.")

        # Analysis state display
        if "analysis" in st.session_state:
            with st.expander("분석 결과 (JSON)"):
                st.json(st.session_state["analysis"])


# ── Helper functions ──────────────────────────────────────────────────────────

def _run_analysis(date_str: str) -> dict:
    """Run all analysis modules and return combined result dict."""
    from modules.data_collector import DataCollector
    from modules import fear_greed, sector_rotation, macro_score, regime_model, qqq_model

    dc = DataCollector()
    data = dc.collect(lookback_years=10)
    market = data["market"]
    fred = data["fred"]
    factors = data["factors"]

    # QQQ return today
    qqq_ret = 0.0
    if "QQQ" in market.columns and len(market["QQQ"].dropna()) >= 2:
        qqq_ret = float(market["QQQ"].dropna().pct_change().iloc[-1] * 100)

    fg = fear_greed.compute(market, fred)
    sector = sector_rotation.compute(market)
    macro = macro_score.compute(fred, market)
    regime = regime_model.predict(market,
                                  dc.compute_factor_returns(factors, market))
    qqq_sig = qqq_model.predict(market, fred, factors)

    return {
        "date": date_str,
        "qqq_return": round(qqq_ret, 4),
        "fear_greed": fg,
        "sector": sector,
        "macro": macro,
        "regime": regime,
        "qqq_signal": qqq_sig,
    }


def _generate(analysis: dict, cfg: dict, date_str: str) -> list[Path]:
    from report.card_generator import CardGenerator

    output_dir = ROOT / cfg.get("analysis", {}).get("output_dir", "output").lstrip("./")
    gen = CardGenerator(cfg["design"], output_dir)
    return gen.generate_all(analysis, date_str)


if __name__ == "__main__":
    main()
