# CLAUDE.md — MHIDSS Project Guide

## 프로젝트 개요
Multi-Horizon Investment Decision Support System.
Macro(FRED) · Fundamental(WRDS) · Technical 3가지 데이터를 통합해
Short/Mid/Long 시계열별 Entry Score(0~100)를 산출한다.

## 실행 방법
```bash
# 의존성 설치
pip install -e ".[dev]"

# 환경 변수 설정
cp .env.example .env
# .env 파일에 FRED_API_KEY, WRDS_USERNAME, WRDS_PASSWORD 입력

# 실행
python main.py run --ticker SPY --format json,html

# 테스트
pytest tests/unit/
pytest tests/integration/  # FRED_API_KEY 필요
```

## 아키텍처
```
config/     - 환경변수, FRED 시리즈 ID, WRDS 필드, 가중치 행렬
data/       - 데이터 fetcher (FRED, WRDS, Technical), Parquet 캐시, 데이터 모델
engine/     - 정규화(MinMax/ZScore/Percentile), 스코어러, 시계열 호라이즌, EntryScoreEngine
utils/      - 날짜 처리, 수학 헬퍼, 검증, 로깅, 재시도
reports/    - JSON/CSV/HTML 리포트 생성
main.py     - CLI 진입점 (typer)
```

## 핵심 원칙
- **Look-ahead bias 금지**: `BaseNormalizer.fit()`은 반드시 `as_of_date` 이전 데이터만 사용
- **불변성**: DataFrame을 직접 수정하지 않고 항상 새 객체 반환
- **추상화 유지**: 데이터 제공자는 `BaseFetcher` 인터페이스를 통해서만 접근
- **부족 데이터**: 0점 처리 금지 → `INSUFFICIENT_DATA` 플래그 후 가중치 재분배

## 가중치 행렬 (config/weights.py)
| 그룹 | Short (1-4W) | Mid (1-6M) | Long (6-24M) |
|------|-------------|------------|--------------|
| Macro | 0.20 | 0.35 | 0.50 |
| Fundamental | 0.10 | 0.30 | 0.45 |
| Technical | 0.70 | 0.35 | 0.05 |

## 커맨드
- `/tdd` — 새 기능 구현 시 TDD 사이클 시작
- `/code-review` — 변경 사항 리뷰
- `/build-fix` — 의존성 또는 타입 에러 수정

## 주요 파일
- `engine/normalizers/base.py` — fit/transform 계약 (아키텍처 핵심)
- `config/weights.py` — 가중치 행렬 (핵심 지식)
- `engine/entry_score.py` — 전체 파이프라인 오케스트레이터
- `data/fetchers/base.py` — 데이터 제공자 교체 인터페이스
- `data/cache/disk_cache.py` — Parquet TTL 캐시
