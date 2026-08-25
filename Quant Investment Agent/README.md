# 퀀트 투자 AI Agent

매일 오전 8시, 사람 개입 없이 시장을 분석하고 TQQQ 진입 판단을 이메일·카카오톡으로 전송하는 퀀트 분석 시스템.

> **⚠️ 투자 권유 아님.** 모든 분석은 참고용 데이터입니다. 투자 결정의 책임은 본인에게 있습니다.

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [파일 구조](#2-파일-구조)
3. [설치 및 환경 설정](#3-설치-및-환경-설정)
4. [실행 방법](#4-실행-방법)
5. [분석 모듈 상세](#5-분석-모듈-상세)
6. [알림 시스템](#6-알림-시스템)
7. [리포트 카드](#7-리포트-카드)
8. [MCP 에이전트 모드](#8-mcp-에이전트-모드)
9. [자동 스케줄링](#9-자동-스케줄링)
10. [설계 원칙](#10-설계-원칙)
11. [알려진 한계](#11-알려진-한계)

---

## 1. 시스템 개요

```
[오전 8시 자동 실행]
       │
       ▼
데이터 수집 ──── yfinance(30년) + FRED + WRDS
       │
       ├── Module 1: QQQ 앙상블 신호    (XGBoost + LightGBM + LSTM + Transformer)
       ├── Module 2: HMM 시장 레짐      (Gaussian HMM + Kalman Filter)
       ├── Module 3: 공포탐욕지수       (7개 컴포넌트, CNN 방식 근사)
       ├── Module 4: 섹터 로테이션      (11개 SPDR ETF 멀티타임프레임 모멘텀)
       ├── Module 5: 매크로 점수        (FRED 5개 지표)
       │
       ├── Claude AI 종합 분석          (모든 신호 취합 → 최종 의견)
       │
       ├── 카드 생성 (7장 PNG, 1080×1080)
       ├── 카카오톡 알림
       └── 인스타그램 업로드 (옵션)
```

---

## 2. 파일 구조

```
투자 에이전트/
├── main.py                   ← 수동 실행 진입점
├── scheduler.py              ← 매일 오전 8시 자동 실행
├── mcp_server.py             ← MCP 서버 (Claude Code 에이전트 모드)
├── config.yaml               ← 카드 디자인 설정
├── .env                      ← API 키 (Git 제외)
├── .mcp.json                 ← Claude Code MCP 자동 등록
├── .gitignore
│
├── modules/
│   ├── data_collector.py     ← yfinance + FRED + WRDS 데이터 수집
│   ├── qqq_model.py          ← TQQQ 장기 진입 신호 (ML/DL 앙상블)
│   ├── regime_model.py       ← HMM 시장 국면 분류
│   ├── fear_greed.py         ← 공포탐욕지수 (7개 컴포넌트)
│   ├── sector_rotation.py    ← 11개 섹터 ETF 로테이션
│   ├── macro_score.py        ← 매크로 환경 점수
│   ├── claude_analyst.py     ← Claude AI 종합 분석
│   └── email_sender.py       ← Gmail SMTP 이메일 발송
│
├── report/
│   └── card_generator.py     ← 인스타그램 카드 7장 PNG 생성
│
├── overseer/
│   ├── main.py               ← 감독관 (독립 프로세스)
│   ├── checkers.py           ← 데이터·모델·레짐 검증
│   └── alerts.py             ← 카카오톡 + 슬랙 알림
│
├── upload/
│   └── instagram.py          ← Instagram Graph API 업로드
│
├── models/                   ← model_YYYYMMDD.pkl 저장
├── cache/                    ← parquet 디스크 캐시
├── output/                   ← report_YYYYMMDD/ 하위 card_01~07.png
└── logs/
    ├── analysis_log.json     ← 누적 실행 로그
    └── scheduler.log
```

---

## 3. 설치 및 환경 설정

### 필수 패키지

```bash
pip install yfinance pandas numpy scikit-learn xgboost lightgbm \
            hmmlearn joblib torch optuna fred requests \
            python-dotenv pyyaml matplotlib pillow
```

### .env 설정

프로젝트 루트의 `.env` 파일에 아래 값을 입력:

```bash
# FRED (무료 발급: fred.stlouisfed.org)
FRED_API_KEY=your_fred_api_key

# WRDS (대학 기관 계정 — 없으면 팩터 피처 스킵됨, 필수 아님)
WRDS_USERNAME=your_username
WRDS_PASSWORD=your_password

# 카카오톡 나에게 보내기
KAKAO_REST_API_KEY=your_rest_api_key
KAKAO_ACCESS_TOKEN=your_access_token      # 약 6시간 유효, 만료 시 자동 갱신
KAKAO_REFRESH_TOKEN=your_refresh_token    # 약 60일 유효

# 이메일 (Gmail 앱 비밀번호 필요)
EMAIL_SENDER=your@gmail.com
EMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx
EMAIL_RECIPIENT=recipient@gmail.com

# Instagram (선택)
INSTAGRAM_ACCESS_TOKEN=your_token
INSTAGRAM_USER_ID=your_user_id

# 캐시·출력 경로
CACHE_DIR=./cache
OUTPUT_DIR=./output
LOG_LEVEL=INFO
```

### 카카오톡 설정 (나에게 보내기)

1. [Kakao Developers](https://developers.kakao.com) → 앱 생성
2. **카카오 로그인 → 동의항목**: `talk_message` **필수 동의** 활성화
3. OAuth 인증 시 반드시 `scope=talk_message` 포함:
   ```
   https://kauth.kakao.com/oauth/authorize?response_type=code
     &client_id={REST_API_KEY}
     &redirect_uri={REDIRECT_URI}
     &scope=talk_message
   ```
4. 발급된 `access_token` / `refresh_token` → `.env` 저장
5. 토큰 만료 시 시스템이 자동으로 갱신하고 `.env` 업데이트

---

## 4. 실행 방법

```bash
# 전체 파이프라인 (업로드 제외, Overseer 없이)
python main.py --no-upload --no-overseer

# 특정 모듈만 테스트
python main.py --module qqq
python main.py --module regime
python main.py --module fear_greed
python main.py --module sector
python main.py --module macro
python main.py --module report      # 더미 데이터로 카드 디자인만 확인

# 모델 강제 재훈련 (월 1회 권장)
python main.py --module qqq --retrain
python main.py --retrain --no-upload

# 자동 실행 등록 (최초 1회 — Windows Task Scheduler)
python scheduler.py --register

# 카카오톡·슬랙 알림 테스트
python overseer/alerts.py --test
```

---

## 5. 분석 모듈 상세

### Module 1: QQQ 앙상블 신호 (`modules/qqq_model.py`)

**목표**: "지금 TQQQ 매수해서 1년 보유하면 좋은 타이밍인가?"

#### 레이블 정의

| 레이블 | 조건 (TQQQ 1년 후 수익률) | 의미 |
|---|---|---|
| **강력매수** | > +50% | 폭락 바닥권, 역사적 대기회 |
| **관망** | 0 ~ +50% | 상승장이나 추가 진입 메리트 낮음 |
| **회피** | < 0% | 하락 위험, TQQQ 보유 부적합 |

#### 데이터

- **기간**: 30년 (1996~현재)
- **TQQQ 합성**: TQQQ 상장 전(~2010.02) 구간은 `QQQ 일일수익률 × 3`으로 합성
- **데이터 소스**: yfinance(시장), FRED(매크로), WRDS(팩터)

#### 피처 엔지니어링 (주요)

| 피처 | 설명 |
|---|---|
| `ath_drawdown` | ATH 대비 현재 낙폭 (핵심 "폭락 감지" 피처) |
| `vix_pct_rank` | VIX의 252일 내 백분위 순위 |
| `qqq_3/6/12m_return` | 다중 기간 모멘텀 |
| `rsi_7/14/21` | 단기·중기 RSI |
| `ma_dev_50/100/200` | 이동평균 이탈률 |
| `macd`, `bb_pct_b` | 기술적 지표 |
| `hyg_chg_20d` | HYG 신용 스프레드 변화 (크레딧 스트레스) |
| `rate_10y`, `yield_spread` | FRED 금리·스프레드 |
| `cpi_yoy`, `unrate` | 인플레이션·실업률 |

#### 모델 구조 (4개 앙상블 + Meta)

```
입력 피처
    ├── XGBoost     (Optuna HPO, 30 trials)
    ├── LightGBM    (Optuna HPO, 30 trials)
    ├── LSTM        (2-layer, hidden=128, dropout=0.3)
    └── Transformer (MultiheadAttention, 4-head)
              │
              └── Meta Learner (Logistic Regression)
                        │
                        └── 최종 신호 + 신뢰도
```

#### 검증 방법

- **5-fold Anchored Walk-Forward CV**
- Fold 경계마다 **252거래일 Purge Zone** 적용 (레이블 겹침 방지)
- Look-ahead bias 완전 차단

#### Analog Finder

오늘 피처 벡터와 가장 유사한 과거 10개 날짜를 찾아 당시 이후 1년/3년 TQQQ 수익률 제시:

```python
result["analogs"]          # TOP 10 유사 과거 구간
result["expected_returns"] # 중앙값·사분위 범위
```

---

### Module 2: 시장 레짐 분류 (`modules/regime_model.py`)

**목표**: HMM으로 현재 시장이 어느 국면인지 판별

#### 관측 변수 (4개)

| 변수 | 설명 |
|---|---|
| SPY 일일수익률 | 시장 방향 |
| VIX (정규화) | 공포 수준 |
| 금리 스프레드 프록시 | TLT 역방향 |
| TLT 수익률 | 채권 흐름 |

#### HMM 구조

- **상태 수**: BIC로 자동 최적 선택 (2~5 탐색)
- **공분산 타입**: full (변수 간 상관 포착)
- **상태 매핑**: SPY 평균 수익률 기준 자동 정렬

| 상태 | 레이블 | 전략 |
|---|---|---|
| 0 | 강세 저변동 (Bull) | QQQ·QLD 비중 확대 |
| 1 | 약세 고변동 (Bear/Correction) | TLT·GLD 헤지, QQQ 축소 |
| 2 | 전환·회복 (Recovery) | 중립, 시그널 확인 후 진입 |

#### Kalman Filter (동적 팩터 가중치)

모멘텀·밸류·퀄리티·사이즈 팩터의 최근 유효성을 실시간 추적.
어떤 팩터가 현재 시장을 주도하는지 동적으로 파악.

---

### Module 3: 공포탐욕지수 (`modules/fear_greed.py`)

**목표**: CNN Fear & Greed Index 방식 근사 구현

#### CNN 대비 백테스트 결과 (2025.05~2026.05)

| 지표 | 값 |
|---|---|
| MAE | **10.0점** |
| 상관계수 | 0.76 |
| 편향(Bias) | **-0.2점** (거의 0) |
| ±10점 이내 | 58% |
| ±20점 이내 | 90% |

> 이전 고정 선형 공식(MAE 20점, 편향 +19.9점) 대비 대폭 개선.

#### 7개 컴포넌트 (모두 롤링 퍼센타일 랭크 기반)

| 컴포넌트 | 원시 지표 | 방향 |
|---|---|---|
| 시장 모멘텀 | SPY / SPY_125MA 이탈률 | 높을수록 탐욕 |
| 주가 강도 | 섹터 ETF 52주 신고가 비율 | 높을수록 탐욕 |
| 주가 폭 | 섹터 ETF 중 50MA 위 비율 | 높을수록 탐욕 |
| 풋/콜 비율 | VIX 5일 MA (역전) | 낮을수록 탐욕 |
| 시장 변동성 | VIX / VIX_50MA (역전) | 낮을수록 탐욕 |
| 안전자산 수요 | TLT - SPY 20일 수익률 차이 (역전) | 낮을수록 탐욕 |
| 정크본드 수요 | HYG - IEF 20일 수익률 차이 | 높을수록 탐욕 |

**핵심 설계**: 고정 선형 변환 대신 **252거래일 롤링 퍼센타일 랭크** 사용.
→ 시장 레짐이 바뀌어도 자동 조정, 편향 거의 0.

| 점수 | 레이블 |
|---|---|
| 0~24 | 극단적 공포 (Extreme Fear) |
| 25~44 | 공포 (Fear) |
| 45~55 | 중립 (Neutral) |
| 56~74 | 탐욕 (Greed) |
| 75~100 | 극단적 탐욕 (Extreme Greed) |

---

### Module 4: 섹터 로테이션 (`modules/sector_rotation.py`)

**목표**: 11개 SPDR 섹터 ETF의 상대 강도 분석

#### 대상 ETF

| 티커 | 섹터 |
|---|---|
| XLK | 기술 (Technology) |
| XLE | 에너지 (Energy) |
| XLF | 금융 (Financials) |
| XLV | 헬스케어 (Health Care) |
| XLI | 산업재 (Industrials) |
| XLY | 임의소비재 (Consumer Discr.) |
| XLP | 필수소비재 (Consumer Staples) |
| XLB | 소재 (Materials) |
| XLRE | 부동산 (Real Estate) |
| XLU | 유틸리티 (Utilities) |
| XLC | 커뮤니케이션 (Communication) |

#### 방법론

- **멀티타임프레임 모멘텀**: 1M(50%), 3M(30%), 6M(20%) 가중 상대 수익률
- **Z-Score 정규화**: 11개 섹터 간 표준화
- **경기 사이클 분류**: 상위 3개 섹터 패턴으로 확장기·후기확장·수축기·회복기 판별
- **로테이션 방향**: 성장형(XLK·XLY·XLC·XLF) vs 방어형(XLP·XLV·XLU·XLRE) 비율

---

### Module 5: 매크로 환경 점수 (`modules/macro_score.py`)

**목표**: FRED 5개 지표로 현재 매크로 환경 0~100점 평가

| 컴포넌트 (각 20점) | 지표 | 점수 기준 |
|---|---|---|
| 금리 방향 | 10Y 금리 20일 변화 | 하락 → 20점 |
| 수익률 곡선 | 10Y-2Y 스프레드 | 양수·확대 → 20점 |
| 달러 강도 | DXY 20일 변화 | 하락 → 20점 |
| 인플레이션 | CPI YoY 수준·추세 | 하락 추세 → 20점 |
| 유동성 | M2 YoY 성장률 | 5%+ → 20점 |

| 점수 | 레이블 |
|---|---|
| 80+ | 위험자산 매우 유리 (Full Risk-On) |
| 60~79 | 위험자산 유리 |
| 40~59 | 중립 |
| ~39 | 위험자산 불리 (Risk-Off) |

---

## 6. 알림 시스템

### 카카오톡 (`overseer/alerts.py`)

- **나에게 보내기** API 사용 (완전 무료)
- 모든 레벨(INFO·WARNING·CRITICAL·SUCCESS) 발송
- **토큰 자동 갱신**: 401 응답 시 refresh_token으로 자동 갱신 후 `.env` 업데이트

#### 발송 시나리오

| 상황 | 레벨 |
|---|---|
| 분석 시작 | INFO |
| 모듈 경고 발생 | WARNING |
| 데이터·모델 이상 감지 | CRITICAL |
| 일일 분석 완료 (상세) | INFO |
| 인스타 업로드 완료 | SUCCESS |

### 슬랙 (선택)

- CRITICAL·SUCCESS 레벨만 발송
- `SLACK_WEBHOOK_URL` 미설정 시 자동 스킵

---

## 7. 리포트 카드

매일 1080×1080 PNG 7장 생성 (`output/report_YYYYMMDD/`):

| 카드 | 내용 |
|---|---|
| Card 1 | 커버 — 날짜, QQQ 수익률, 오늘의 신호, 공포탐욕 요약 |
| Card 2 | QQQ 타이밍 신호 — 앙상블 결과, 신뢰도 게이지, 개별 모델 의견, 주요 피처 |
| Card 3 | 공포탐욕지수 — 점수, 7개 컴포넌트 바 차트 |
| Card 4 | 시장 레짐 — HMM 상태 파이 차트, 팩터 가중치 바 차트 |
| Card 5 | 섹터 로테이션 — 11개 ETF Z-Score 차트, 강세·약세 TOP3 |
| Card 6 | 매크로 환경 — 점수, 5개 지표 레이더 차트 |
| Card 7 | 면책 고지 — 데이터 출처, 모델 목록, 투자 권유 아님 안내 |

디자인 커스터마이징: `config.yaml`에서 색상·폰트 조정
디자인 편집기: `streamlit run report/editor.py`

---

## 8. MCP 에이전트 모드

이 폴더에서 Claude Code를 열면 `.mcp.json`이 자동으로 MCP 서버를 등록.
Claude가 직접 어떤 도구를 언제 쓸지 결정하는 에이전트 모드로 전환.

```
사용자: "오늘 TQQQ 사도 돼?"
Claude: [collect_data 호출] → [get_qqq_signal 호출] → [get_regime 호출]
        → 스스로 신호 취합 후 최종 의견 제시
```

#### MCP 도구 목록

| 도구 | 기능 |
|---|---|
| `collect_data` | 시장 데이터 수집 |
| `get_qqq_signal` | TQQQ 진입 신호 + Analog |
| `get_regime` | HMM 국면 분류 |
| `get_fear_greed` | 공포탐욕지수 |
| `get_sector_rotation` | 섹터 분석 |
| `get_macro_score` | 매크로 점수 |
| `generate_report_cards` | 카드 7장 생성 |
| `send_email_report` | 이메일 발송 |

---

## 9. 자동 스케줄링

### Windows Task Scheduler 등록

```bash
python scheduler.py --register   # 최초 1회 실행
```

등록 후 매일 오전 8시에 자동 실행. 수동 개입 불필요.

### 실행 흐름

```
[오전 8시]
   └── scheduler.py
          └── main.py --no-overseer
                 ├── 데이터 수집
                 ├── 5개 분석 모듈
                 ├── Claude 종합 분석
                 ├── 카드 생성
                 ├── 카카오톡 알림
                 └── 인스타그램 업로드
```

### 모델 재훈련 (월 1회 권장)

```bash
python main.py --module qqq --retrain
```

---

## 10. 설계 원칙

### Look-ahead Bias 완전 차단

```python
# 레이블 생성
fwd_ret = tqqq.pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

# Walk-forward: fold 경계마다 252일 Purge Zone 적용
test_start = train_end + 252
```

### 에러 격리

모듈 하나 실패해도 전체 파이프라인 중단 없음:

```python
result = _safe_run(lambda: module.compute(...), "module_name", fallback={})
```

### 캐시 정책

- 당일 중복 API 호출 차단 (parquet 캐시)
- 캐시 경로: `./cache/` (`.env`의 `CACHE_DIR`)

### 토큰 보안

- `.env`는 `.gitignore`에 포함 — GitHub 절대 업로드 금지
- 카카오 토큰: 만료 시 자동 갱신 후 `.env` 즉시 업데이트

---

## 11. 알려진 한계

| 항목 | 현황 |
|---|---|
| Put/Call 비율 | 실제 CBOE 데이터 없음 → VIX 5일 MA로 대체 |
| WRDS 팩터 | 비밀번호 미설정 시 `earnings_surprise`, `quality_factor`가 0으로 채워짐 |
| 공포탐욕 백테스트 | CNN API가 최근 1년치만 제공 → 샘플 50개 한계 |
| Analog 유사도 | 15~20% 수준 (30년 피처 공간이 넓음, 절대값보다 순위 기준으로 해석) |
| Claude 분석 | 120초 타임아웃, 초과 시 규칙 기반 fallback으로 대체 |
| 인스타그램 업로드 | 토큰 미설정 시 자동 스킵 |

---

*마지막 업데이트: 2026-05-03*
