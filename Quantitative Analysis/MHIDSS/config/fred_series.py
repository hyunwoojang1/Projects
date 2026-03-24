"""FRED 시리즈 ID 상수 레지스트리."""

# 금리
FEDFUNDS = "FEDFUNDS"       # 연방기금금리
DGS10 = "DGS10"             # 10년 국채 수익률
DGS2 = "DGS2"               # 2년 국채 수익률

# 인플레이션
CPIAUCSL = "CPIAUCSL"       # 소비자물가지수 (CPI)
PCEPILFE = "PCEPILFE"       # 근원 PCE 물가

# 고용
UNRATE = "UNRATE"           # 실업률
ICSA = "ICSA"               # 신규 실업수당 청구건수

# 유동성
M2SL = "M2SL"               # M2 통화량

# 신용
BAA = "BAA"                 # Moody's BAA 등급 회사채 수익률
AAA = "AAA"                 # Moody's AAA 등급 회사채 수익률

# 파생 지표 (계산 필요)
YIELD_CURVE_SPREAD = "YIELD_CURVE_SPREAD"   # DGS10 - DGS2
CREDIT_SPREAD = "CREDIT_SPREAD"             # BAA - AAA

# 전체 수집 대상 시리즈 (파생 제외)
FETCH_SERIES: list[str] = [
    FEDFUNDS, DGS10, DGS2,
    CPIAUCSL, PCEPILFE,
    UNRATE, ICSA,
    M2SL,
    BAA, AAA,
]

# 파생 지표 정의 (key: 파생 ID, value: (series_a, series_b, operation))
DERIVED_SERIES: dict[str, tuple[str, str, str]] = {
    YIELD_CURVE_SPREAD: (DGS10, DGS2, "subtract"),
    CREDIT_SPREAD: (BAA, AAA, "subtract"),
}
