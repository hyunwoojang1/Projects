"""WRDS Compustat 테이블/필드 레지스트리."""

# ── 테이블 경로 ───────────────────────────────────────────────────────────────
FUNDA_TABLE   = "comp.funda"            # 연간 재무제표
COMPANY_TABLE = "comp.company"          # 기업 헤더 (gsector, tic 포함)
FUNDQ_TABLE   = "comp.fundq"            # 분기 재무제표
SECD_TABLE    = "comp.g_secd"           # 일별 증권 데이터
CRSP_DSF      = "crsp.dsf"             # CRSP 일별 주식 파일
LINK_TABLE    = "crsp.ccmxpf_lnkhist"  # PERMNO ↔ GVKEY 연결 테이블

# ── 공통 식별자 필드 (comp.funda) ─────────────────────────────────────────────
KEY_FIELDS = ["f.gvkey", "f.datadate", "f.fyear", "f.indfmt", "f.consol", "f.popsrc", "f.datafmt", "f.tic"]

# ── 섹터 필드 (comp.company JOIN) — tic은 comp.funda에 있어서 KEY_FIELDS에 포함 ──
COMPANY_FIELDS = ["c.gsector"]   # GICS 섹터코드

# ── 재무 지표 필드 (comp.funda) ───────────────────────────────────────────────
FINANCIAL_FIELDS = [
    "f.prcc_f",   # 회계연도 말 주가
    "f.csho",     # 발행 주식 수 (백만 주)
    "f.ceq",      # 주주자본 (장부가)
    "f.ni",       # 순이익
    "f.epsfx",    # 희석 EPS
    "f.oancf",    # 영업활동 현금흐름
    "f.capx",     # 자본적 지출
    "f.mkvalt",   # 시가총액
    "f.dltt",     # 장기 부채
    "f.dlc",      # 단기 부채
    "f.sale",     # 매출액
]

# ── 파생 지표 계산 정의 ───────────────────────────────────────────────────────
DERIVED_FIELDS: dict[str, tuple[str, str]] = {
    "pbr":             ("prcc_f", "ceq / csho"),
    "roe":             ("ni", "ceq"),
    "eps_change_rate": ("epsfx - epsfx_lag4", "abs(epsfx_lag4)"),
    "fcf_yield":       ("oancf - capx", "mkvalt"),
    "de_ratio":        ("dltt + dlc", "ceq"),
    "revenue_growth":  ("sale - sale_lag4", "sale_lag4"),
    "earnings_yield":  ("epsfx", "prcc_f"),
}

# ── 필터 조건 (Compustat 표준 필터) ──────────────────────────────────────────
STANDARD_FILTERS = {
    "indfmt":  "INDL",
    "consol":  "C",
    "popsrc":  "D",
    "datafmt": "STD",
}
