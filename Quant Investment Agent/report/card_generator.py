"""Report Generator — 7장의 1080x1080 인스타그램 카드 PNG 생성.

matplotlib로 차트를 그리고 Pillow로 합성.
design_config(config.yaml에서 로드)로 모든 색상/폰트를 제어.
"""

from __future__ import annotations

import io
import math
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

# ── matplotlib 한글 폰트 설정 (Windows: Malgun Gothic) ────────────────────────
def _setup_mpl_korean() -> None:
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/NanumGothic.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            prop = fm.FontProperties(fname=path)
            matplotlib.rcParams["font.family"] = prop.get_name()
            break
    matplotlib.rcParams["axes.unicode_minus"] = False  # 마이너스 깨짐 방지

_setup_mpl_korean()
import numpy as np
from matplotlib.figure import Figure
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")  # headless rendering

W = H = 1080

# ── Font resolution ───────────────────────────────────────────────────────────

def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/malgun.ttf",       # 맑은 고딕 (Windows)
        "C:/Windows/Fonts/NanumGothic.ttf",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # macOS
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


# ── Color helpers ─────────────────────────────────────────────────────────────

def _hex(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    return tuple(int(c[i:i+2], 16) for i in (0, 2, 4))


def _mpl_color(hex_str: str) -> str:
    return hex_str


# ── Card generator ────────────────────────────────────────────────────────────

class CardGenerator:
    def __init__(self, design: dict, output_dir: Path) -> None:
        self.d = design
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public interface ───────────────────────────────────────────────────────

    def generate_all(self, analysis: dict, date_str: str | None = None) -> list[Path]:
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        out_dir = self.output_dir / f"report_{date_str.replace('-', '')}"
        out_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        generators = [
            ("card_01.png", self._card_01_cover),
            ("card_02.png", self._card_02_timing),
            ("card_03.png", self._card_03_fear_greed),
            ("card_04.png", self._card_04_regime),
            ("card_05.png", self._card_05_sector),
            ("card_06.png", self._card_06_macro),
            ("card_07.png", self._card_07_disclaimer),
        ]

        for filename, gen_fn in generators:
            try:
                img = gen_fn(analysis, date_str)
                path = out_dir / filename
                img.save(path, "PNG", quality=95)
                paths.append(path)
            except Exception as e:
                print(f"[WARN] {filename} 생성 실패: {e}")

        return paths

    # ── Card 1: 표지 ───────────────────────────────────────────────────────────

    def _card_01_cover(self, analysis: dict, date_str: str) -> Image.Image:
        img, draw = self._base_card()

        # Date + day
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        days_kr = ["월", "화", "수", "목", "금", "토", "일"]
        date_label = f"{dt.strftime('%Y.%m.%d')} ({days_kr[dt.weekday()]})"
        self._text(draw, date_label, (W // 2, 140), size="small",
                   color=self.d["text_secondary"], anchor="mm")

        # Logo
        self._text(draw, self.d.get("logo_text", "QUANT SIGNAL"),
                   (W // 2, 200), size="subtitle",
                   color=self.d["accent_color"], anchor="mm")

        # QQQ return
        qqq_ret = analysis.get("qqq_return", 0.0)
        ret_color = self.d["positive_color"] if qqq_ret >= 0 else self.d["negative_color"]
        self._text(draw, "QQQ", (W // 2, 420), size="subtitle",
                   color=self.d["text_secondary"], anchor="mm")
        self._text(draw, f"{qqq_ret:+.2f}%", (W // 2, 520), size="title",
                   color=ret_color, anchor="mm")

        # Signal
        qqq = analysis.get("qqq_signal", {})
        signal_label = qqq.get("signal_label", "—")
        self._text(draw, f"오늘의 신호: {signal_label}", (W // 2, 680),
                   size="body", color=self.d["text_primary"], anchor="mm")

        # Fear & Greed quick view
        fg_score = analysis.get("fear_greed", {}).get("score", 50)
        fg_label = analysis.get("fear_greed", {}).get("label", "—")
        self._text(draw, f"공포탐욕지수 {fg_score}  |  {fg_label}",
                   (W // 2, 780), size="small", color=self.d["text_secondary"],
                   anchor="mm")

        self._footer(draw, date_str)
        return img

    # ── Card 2: QQQ 타이밍 신호 ───────────────────────────────────────────────

    def _card_02_timing(self, analysis: dict, date_str: str) -> Image.Image:
        img, draw = self._base_card()
        qqq = analysis.get("qqq_signal", {})

        self._card_title(draw, "📊 QQQ 타이밍 신호")

        signal_label = qqq.get("signal_label", "QQQ (보유)")
        conf = qqq.get("confidence", 0.5)
        low_conf = qqq.get("low_confidence", False)

        # Main signal
        signal_color = self._signal_color(signal_label)
        self._text(draw, signal_label, (W // 2, 380), size="subtitle",
                   color=signal_color, anchor="mm")

        # Confidence gauge (matplotlib → embed)
        gauge_img = self._gauge_chart(conf * 100, label="신뢰도",
                                      color=signal_color)
        img.paste(gauge_img, (W // 2 - 160, 420))

        # Star rating
        stars = self._stars(conf)
        self._text(draw, stars, (W // 2, 640), size="body",
                   color=self.d["accent_color"], anchor="mm")

        if low_conf:
            self._text(draw, "⚠ 신뢰도 낮음 — 추가 확인 권장",
                       (W // 2, 700), size="small",
                       color=self.d["secondary_color"], anchor="mm")

        # Individual model votes
        models = qqq.get("models", {})
        y_pos = 760
        for name, pred in models.items():
            self._text(draw, f"{name.upper()}: {pred}",
                       (W // 2, y_pos), size="small",
                       color=self.d["text_secondary"], anchor="mm")
            y_pos += 50

        # Top features
        top_feats = qqq.get("top_features", [])
        if top_feats:
            feat_text = " | ".join(
                f"{f['name']} {f['importance']*100:.1f}%"
                for f in top_feats[:3])
            self._text(draw, f"주요 피처: {feat_text}",
                       (W // 2, y_pos + 20), size="label",
                       color=self.d["text_secondary"], anchor="mm")

        self._footer(draw, date_str)
        return img

    # ── Card 3: 공포탐욕지수 ──────────────────────────────────────────────────

    def _card_03_fear_greed(self, analysis: dict, date_str: str) -> Image.Image:
        img, draw = self._base_card()
        fg = analysis.get("fear_greed", {})

        self._card_title(draw, "😰 공포탐욕지수")

        score = fg.get("score", 50)
        label = fg.get("label", "중립")
        change = fg.get("change")

        # Big score
        score_color = self._fg_color(score)
        self._text(draw, str(score), (W // 2, 380), size="title",
                   color=score_color, anchor="mm")
        self._text(draw, label, (W // 2, 460), size="body",
                   color=score_color, anchor="mm")

        if change is not None:
            arrow = "▲" if change > 0 else "▼"
            chg_color = (self.d["positive_color"] if change > 0
                         else self.d["negative_color"])
            self._text(draw, f"지난주 대비 {arrow}{abs(change)}",
                       (W // 2, 510), size="small", color=chg_color,
                       anchor="mm")

        # Component bar chart
        components = fg.get("components", {})
        if components:
            bar_img = self._fg_components_chart(components)
            img.paste(bar_img, (60, 550))

        self._footer(draw, date_str)
        return img

    # ── Card 4: 시장 레짐 & 팩터 ──────────────────────────────────────────────

    def _card_04_regime(self, analysis: dict, date_str: str) -> Image.Image:
        img, draw = self._base_card()
        regime = analysis.get("regime", {})

        self._card_title(draw, "🧭 시장 레짐 & 팩터 환경")

        state_label = regime.get("state_label", "알 수 없음")
        self._text(draw, state_label, (W // 2, 300), size="body",
                   color=self.d["accent_color"], anchor="mm")

        desc = regime.get("regime_description", "")
        self._text(draw, desc, (W // 2, 360), size="small",
                   color=self.d["text_secondary"], anchor="mm")

        # Regime probability pie chart
        state_probs = regime.get("state_probs", {})
        if state_probs:
            pie_img = self._pie_chart(state_probs)
            img.paste(pie_img, (60, 400))

        # Factor weight bar chart
        fw = regime.get("factor_weights", {})
        if fw:
            factor_img = self._factor_bar_chart(fw)
            img.paste(factor_img, (560, 400))

        strategy = regime.get("strategy", "")
        wrapped = textwrap.fill(strategy, width=30)
        y = 870
        for line in wrapped.split("\n"):
            self._text(draw, line, (W // 2, y), size="small",
                       color=self.d["text_primary"], anchor="mm")
            y += 40

        self._footer(draw, date_str)
        return img

    # ── Card 5: 섹터 로테이션 ─────────────────────────────────────────────────

    def _card_05_sector(self, analysis: dict, date_str: str) -> Image.Image:
        img, draw = self._base_card()
        sector = analysis.get("sector", {})

        self._card_title(draw, "🔄 섹터 로테이션")

        rot_dir = sector.get("rotation_dir", "—")
        self._text(draw, rot_dir, (W // 2, 290), size="small",
                   color=self.d["accent_color"], anchor="mm")

        cycle = sector.get("cycle_phase", "—")
        self._text(draw, f"경기 사이클: {cycle}", (W // 2, 340),
                   size="small", color=self.d["text_secondary"], anchor="mm")

        # Heatmap
        ranked = sector.get("ranked", [])
        if ranked:
            heatmap = self._sector_heatmap(ranked)
            img.paste(heatmap, (60, 380))

        # TOP 3 / BOTTOM 3
        top3 = sector.get("top3", [])
        bottom3 = sector.get("bottom3", [])
        self._text(draw, f"▲ 강세: {', '.join(top3)}",
                   (W // 2, 900), size="small",
                   color=self.d["positive_color"], anchor="mm")
        self._text(draw, f"▼ 약세: {', '.join(bottom3)}",
                   (W // 2, 950), size="small",
                   color=self.d["negative_color"], anchor="mm")

        self._footer(draw, date_str)
        return img

    # ── Card 6: 매크로 환경 ───────────────────────────────────────────────────

    def _card_06_macro(self, analysis: dict, date_str: str) -> Image.Image:
        img, draw = self._base_card()
        macro = analysis.get("macro", {})

        self._card_title(draw, "🌐 매크로 환경")

        score = macro.get("score", 50)
        macro_label = macro.get("label", "—")
        score_color = self._macro_color(score)

        self._text(draw, f"{score}점", (W // 2, 310), size="subtitle",
                   color=score_color, anchor="mm")
        self._text(draw, macro_label, (W // 2, 380), size="small",
                   color=score_color, anchor="mm")

        # Radar chart
        components = macro.get("components", {})
        if components:
            radar = self._radar_chart(components)
            img.paste(radar, (W // 2 - 240, 420))

        # Component detail
        details = macro.get("details", {})
        y = 850
        for key, info in list(details.items())[:5]:
            desc = info.get("desc", "") if isinstance(info, dict) else ""
            self._text(draw, desc, (W // 2, y), size="label",
                       color=self.d["text_secondary"], anchor="mm")
            y += 40

        self._footer(draw, date_str)
        return img

    # ── Card 7: 면책 고지 ─────────────────────────────────────────────────────

    def _card_07_disclaimer(self, analysis: dict, date_str: str) -> Image.Image:
        img, draw = self._base_card()

        self._card_title(draw, "⚠️ 면책 고지")

        lines = [
            "본 콘텐츠는 투자 권유가 아닙니다.",
            "모든 투자 결정은 본인의 책임입니다.",
            "",
            "데이터 출처:",
            "• 시장 데이터: Yahoo Finance (yfinance)",
            "• 거시 데이터: FRED (Federal Reserve)",
            "• 팩터 데이터: WRDS / Compustat",
            "",
            "모델: XGBoost, LightGBM, LSTM, Transformer",
            "백테스트 성과는 미래 수익을 보장하지 않습니다.",
            "",
            "팔로우하고 매일 시장 분석을 받아보세요.",
        ]

        y = 350
        for line in lines:
            color = (self.d["accent_color"] if line.startswith("•")
                     else self.d["text_primary"] if line
                     else self.d["text_secondary"])
            self._text(draw, line, (W // 2, y), size="small",
                       color=color, anchor="mm")
            y += 55

        self._footer(draw, date_str)
        return img

    # ── Chart helpers ─────────────────────────────────────────────────────────

    def _gauge_chart(self, value: float, label: str, color: str) -> Image.Image:
        fig, ax = plt.subplots(figsize=(3.2, 1.6),
                               facecolor=self.d["background_color"])
        ax.set_facecolor(self.d["background_color"])

        theta = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), color="#333333", linewidth=8)

        # Fill arc
        fill_theta = np.linspace(0, np.pi * value / 100, 100)
        ax.plot(np.cos(fill_theta), np.sin(fill_theta),
                color=color, linewidth=8)

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.1, 1.2)
        ax.axis("off")
        ax.text(0, -0.05, f"{value:.0f}%", ha="center", va="top",
                fontsize=18, color=color, fontweight="bold")
        ax.text(0, 1.1, label, ha="center", va="bottom",
                fontsize=11, color=self.d["text_secondary"])

        return self._fig_to_pil(fig, size=(320, 160))

    def _fg_components_chart(self, components: dict) -> Image.Image:
        names_kr = {
            "market_momentum": "시장 모멘텀",
            "price_strength": "주가 강도",
            "price_breadth": "주가 폭",
            "put_call": "풋/콜 비율",
            "volatility": "시장 변동성",
            "safe_haven": "안전자산 수요",
            "junk_bond": "정크본드 수요",
        }

        labels = [names_kr.get(k, k) for k in components]
        values = list(components.values())
        colors = [self._fg_color(v) for v in values]

        fig, ax = plt.subplots(figsize=(9.6, 4.0),
                               facecolor=self.d["card_bg"])
        ax.set_facecolor(self.d["card_bg"])

        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, values, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=10, color=self.d["text_primary"])
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", colors=self.d["text_secondary"], labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#333333")

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val}", va="center", fontsize=9,
                    color=self.d["text_primary"])

        fig.tight_layout(pad=0.5)
        return self._fig_to_pil(fig, size=(960, 400))

    def _pie_chart(self, state_probs: dict) -> Image.Image:
        labels = list(state_probs.keys())
        sizes = list(state_probs.values())
        colors = [self.d["positive_color"],
                  self.d["negative_color"],
                  self.d["neutral_color"]][:len(labels)]

        fig, ax = plt.subplots(figsize=(4.5, 4.5),
                               facecolor=self.d["card_bg"])
        ax.set_facecolor(self.d["card_bg"])
        wedges, texts, autotexts = ax.pie(
            sizes, labels=None, autopct="%1.0f%%",
            colors=colors[:len(sizes)],
            wedgeprops={"linewidth": 1, "edgecolor": self.d["card_bg"]},
            textprops={"color": self.d["text_primary"], "fontsize": 9},
        )
        for at in autotexts:
            at.set_color(self.d["background_color"])
            at.set_fontsize(10)

        ax.legend(wedges, [l[:10] for l in labels],
                  loc="lower center", ncol=1,
                  fontsize=8, framealpha=0,
                  labelcolor=self.d["text_secondary"])
        return self._fig_to_pil(fig, size=(450, 430))

    def _factor_bar_chart(self, factor_weights: dict) -> Image.Image:
        names_kr = {
            "momentum": "모멘텀", "value": "밸류",
            "quality": "퀄리티", "size": "사이즈",
        }
        labels = [names_kr.get(k, k) for k in factor_weights]
        values = [v * 100 for v in factor_weights.values()]

        fig, ax = plt.subplots(figsize=(4.5, 4.5),
                               facecolor=self.d["card_bg"])
        ax.set_facecolor(self.d["card_bg"])

        colors = [self.d["accent_color"], self.d["secondary_color"],
                  "#8888FF", "#FFD700"]
        ax.bar(labels, values, color=colors[:len(labels)], width=0.5)
        ax.set_ylim(0, 60)
        ax.set_ylabel("가중치 (%)", color=self.d["text_secondary"], fontsize=9)
        ax.tick_params(colors=self.d["text_secondary"], labelsize=9)
        for sp in ax.spines.values():
            sp.set_color("#333333")
        ax.set_facecolor(self.d["card_bg"])

        for i, v in enumerate(values):
            ax.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9,
                    color=self.d["text_primary"])

        return self._fig_to_pil(fig, size=(450, 430))

    def _sector_heatmap(self, ranked: list) -> Image.Image:
        tickers = [r[0] for r in ranked]
        zscores = [r[2] for r in ranked]

        fig, ax = plt.subplots(figsize=(9.6, 5.5),
                               facecolor=self.d["card_bg"])
        ax.set_facecolor(self.d["card_bg"])

        colors = ["#CC2222" if z < -0.5 else
                  "#228822" if z > 0.5 else
                  "#888822" for z in zscores]

        bars = ax.bar(tickers, zscores, color=colors, width=0.6)
        ax.axhline(0, color="#555555", linewidth=1)
        ax.set_ylabel("Z-Score", color=self.d["text_secondary"], fontsize=9)
        ax.tick_params(axis="x", colors=self.d["text_primary"], labelsize=9, rotation=30)
        ax.tick_params(axis="y", colors=self.d["text_secondary"], labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#333333")

        for bar, z in zip(bars, zscores):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    z + (0.05 if z >= 0 else -0.12),
                    f"{z:.2f}",
                    ha="center", fontsize=7, color=self.d["text_secondary"])

        fig.tight_layout(pad=0.5)
        return self._fig_to_pil(fig, size=(960, 500))

    def _radar_chart(self, components: dict) -> Image.Image:
        names_kr = {
            "rate_direction": "금리 방향", "yield_curve": "수익률 곡선",
            "dollar": "달러 강도", "inflation": "인플레이션", "liquidity": "유동성",
        }
        labels = [names_kr.get(k, k) for k in components]
        values = [v for v in components.values()]
        n = len(labels)

        if n == 0:
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            return self._fig_to_pil(fig, size=(480, 480))

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        values_plot = values + [values[0]]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(4.8, 4.8),
                               subplot_kw={"polar": True},
                               facecolor=self.d["card_bg"])
        ax.set_facecolor(self.d["card_bg"])

        ax.plot(angles, values_plot, color=self.d["accent_color"], linewidth=2)
        ax.fill(angles, values_plot, alpha=0.25, color=self.d["accent_color"])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, size=9, color=self.d["text_primary"])
        ax.set_ylim(0, 20)
        ax.set_yticks([5, 10, 15, 20])
        ax.set_yticklabels(["5", "10", "15", "20"],
                           size=7, color=self.d["text_secondary"])
        ax.tick_params(colors=self.d["text_secondary"])
        ax.spines["polar"].set_color("#333333")
        ax.grid(color="#333333", alpha=0.5)

        return self._fig_to_pil(fig, size=(480, 480))

    # ── Base card & text utilities ────────────────────────────────────────────

    def _base_card(self) -> tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("RGB", (W, H), _hex(self.d["background_color"]))
        draw = ImageDraw.Draw(img)
        # Accent line at top
        draw.rectangle([0, 0, W, 6], fill=_hex(self.d["accent_color"]))
        return img, draw

    def _card_title(self, draw: ImageDraw.ImageDraw, title: str) -> None:
        self._text(draw, title, (W // 2, 180), size="body",
                   color=self.d["text_primary"], anchor="mm")
        draw.rectangle([60, 215, W - 60, 218],
                       fill=_hex(self.d["accent_color"] + "88"))

    def _footer(self, draw: ImageDraw.ImageDraw, date_str: str) -> None:
        draw.rectangle([0, H - 5, W, H], fill=_hex(self.d["accent_color"]))
        self._text(draw, f"© {date_str} | 투자 권유 아님",
                   (W // 2, H - 30), size="label",
                   color=self.d["text_secondary"], anchor="mm")
        logo = self.d.get("logo_text", "QUANT SIGNAL")
        self._text(draw, logo, (W - 80, H - 30), size="label",
                   color=self.d["accent_color"], anchor="mm")

    def _text(self, draw: ImageDraw.ImageDraw, text: str,
              pos: tuple[int, int], size: str = "body",
              color: str = "#FFFFFF", anchor: str = "mm") -> None:
        size_map = {
            "title": self.d.get("font_size_title", 72),
            "subtitle": self.d.get("font_size_subtitle", 48),
            "body": self.d.get("font_size_body", 36),
            "small": self.d.get("font_size_small", 28),
            "label": self.d.get("font_size_label", 24),
        }
        font = _load_font(size_map.get(size, 32))
        draw.text(pos, text, font=font, fill=_hex(color), anchor=anchor)

    def _fig_to_pil(self, fig: Figure, size: tuple[int, int]) -> Image.Image:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        return img.resize(size, Image.LANCZOS)

    # ── Color helpers ──────────────────────────────────────────────────────────

    def _fg_color(self, score: float) -> str:
        if score < 25:
            return "#FF2222"
        if score < 45:
            return "#FF8C00"
        if score < 56:
            return "#FFD700"
        if score < 75:
            return "#88CC44"
        return "#00FF88"

    def _macro_color(self, score: int) -> str:
        if score >= 60:
            return self.d["positive_color"]
        if score >= 40:
            return self.d["neutral_color"]
        return self.d["negative_color"]

    def _signal_color(self, label: str) -> str:
        if "TQQQ" in label:
            return self.d["accent_color"]
        if "QLD" in label:
            return "#88FF44"
        if "현금" in label or "Cash" in label:
            return self.d["negative_color"]
        return self.d["text_primary"]

    def _stars(self, confidence: float) -> str:
        filled = round(confidence * 5)
        return "★" * filled + "☆" * (5 - filled)
