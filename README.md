# LottoSight

ระบบวิเคราะห์และทำนายสลากกินแบ่งรัฐบาลไทย โดยใช้ข้อมูลย้อนหลังตั้งแต่ปี 2007 ผ่าน Machine Learning

> **หมายเหตุ:** ลอตเตอรี่เป็นเกมความน่าจะเป็นแบบสุ่ม ระบบนี้สร้างขึ้นเพื่อการศึกษาด้าน Data Engineering และ Machine Learning ไม่ใช่การรับประกันผลการทำนาย

---

## Features

- ดึงข้อมูลผลสลากย้อนหลังตั้งแต่ปี 2007 อัตโนมัติ
- วิเคราะห์สถิติและ pattern ของเลขที่ออก
- Train โมเดล ML หลายประเภท พร้อม walk-forward validation
- ทำนาย top-15 candidates สำหรับทุก target (เลขท้าย 2 ตัว, 3 ตัว, เลขหน้า 3 ตัว, รางวัลที่ 1)
- ดึงผลงวดใหม่และ retrain อัตโนมัติทุกวันที่ 2 และ 17 ของเดือน

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Database | PostgreSQL 16 |
| ORM / Migration | SQLAlchemy 2.0, Alembic |
| Data Collection | requests, BeautifulSoup |
| ML | LightGBM, PyTorch (LSTM), scikit-learn |
| Experiment Tracking | MLflow |
| Scheduler | APScheduler |
| Containerization | Docker Compose |

---

## Models

| Model | คำอธิบาย |
|-------|---------|
| Baseline Random | สุ่ม uniform — ใช้เป็น baseline |
| Baseline Frequency | weighted ตามความถี่ที่เคยออก |
| Statistical | Gap + Frequency score |
| LightGBM | Binary classification บน engineered features |
| LSTM | Sequential model จาก PyTorch |
| Ensemble | Weighted average ของทุก model |

---

## Getting Started

### 1. Prerequisites

- Python 3.11+
- Docker & Docker Compose

### 2. Clone และติดตั้ง

```bash
git clone https://github.com/<your-username>/lottosight.git
cd lottosight

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. ตั้งค่า environment

```bash
cp .env.example .env
# แก้ไข .env ตามต้องการ
```

### 4. Start services

```bash
docker compose up -d
```

### 5. สร้าง database

```bash
alembic upgrade head
```

---

## Usage

### Bootstrap ข้อมูลย้อนหลัง (ครั้งแรกครั้งเดียว)

```bash
python main.py bootstrap
```

### Train โมเดล

```bash
# train ทุก target
python main.py train --target all

# หรือเลือก target เดียว
python main.py train --target back2
```

Target ที่รองรับ: `back2`, `back3`, `front3`, `prize1_last2`

### ดูคำทำนายงวดถัดไป

```bash
python main.py predict
```

### รัน scheduler อัตโนมัติ

```bash
python main.py serve
```

ระบบจะดึงผล → retrain → predict เองทุกวันที่ 2 และ 17 เวลา 09:00 (Bangkok time)

### รันทุกอย่างในคำสั่งเดียว

```bash
python main.py
```

---

## Project Structure

```
lottosight/
├── config/          # pydantic-settings configuration
├── db/              # SQLAlchemy models + Alembic migrations
├── scraper/         # Data collection (GLO API, rayriffy, BeautifulSoup, GitHub archive)
├── etl/             # Extract → Transform → Validate → Load
├── features/        # Feature engineering
├── models/          # ML models (baseline, statistical, lgbm, lstm, ensemble)
├── pipeline/        # Training, prediction, evaluation pipelines
├── scheduler/       # APScheduler jobs
├── tests/
├── docker-compose.yml
├── requirements.txt
└── main.py
```

---

## Data Sources

| Source | ใช้สำหรับ |
|--------|---------|
| [vicha-w/thai-lotto-archive](https://github.com/vicha-w/thai-lotto-archive) | Bootstrap ข้อมูลย้อนหลัง 2007-ปัจจุบัน |
| GLO Official API (glo.or.th) | ดึงผลงวดล่าสุด |
| rayriffy Thai Lotto API | Fallback |
| glo.or.th (HTML) | Last resort fallback |

---

## Evaluation Metrics

| Metric | คำอธิบาย |
|--------|---------|
| hit_top1 | ถูกใน rank 1 |
| hit_top5 | ถูกใน top-5 |
| hit_top10 | ถูกใน top-10 |
| MRR | Mean Reciprocal Rank |

> Random baseline สำหรับ back2 (100 candidates) คือ hit_top10 = 10%

---

## License

MIT
