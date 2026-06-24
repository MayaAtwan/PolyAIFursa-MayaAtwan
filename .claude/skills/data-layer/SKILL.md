---
name: yolo-api-data-layer
description: >-
  Use this skill for ANY task that touches the YOLO service database layer — always use
  it even if the user doesn't mention SQLAlchemy explicitly. Trigger on: "add an endpoint",
  "add a column", "add a table", "refactor the api", "write tests for", "the database layer",
  "delete a prediction", "filter by", "recent sessions", "store", "persist", "query", or
  any change to services/yolo/ that involves reading or writing data. When in doubt, use
  this skill — it contains all schema, ORM patterns, test isolation rules, and hard
  constraints for the YOLO FastAPI service. Do NOT proceed with any database-layer task
  without reading this file first.
---

# yolo-api-data-layer

> **STOP. Read this entire file before touching any code, running any command, or exploring
> the project.** This file contains everything you need — schema, patterns, rules, and test
> isolation. Do not substitute your own judgment for what is written here.

---

## Project structure

```
services/yolo/
├── app.py           ← FastAPI endpoints — each DB endpoint uses Depends(get_db)
├── models.py        ← SQLAlchemy ORM models
├── db.py            ← engine, SessionLocal, get_db()
├── requirements.txt ← must include sqlalchemy>=2.0, psycopg2-binary>=2.9
└── tests/
    └── conftest.py  ← test DB isolation via dependency_overrides
```

Read `models.py`, `db.py`, `app.py`, and `tests/conftest.py` before making changes.

---

## Schema

### prediction_sessions
| Column          | Type         | Notes                     |
|-----------------|--------------|---------------------------|
| uid             | Text, PK     | UUID string               |
| timestamp       | DateTime     | server_default=func.now() |
| original_image  | Text         | filesystem path           |
| predicted_image | Text         | filesystem path           |

### detection_objects
| Column         | Type         | Notes                                  |
|----------------|--------------|----------------------------------------|
| id             | Integer, PK  | autoincrement=True                     |
| prediction_uid | Text, FK     | → prediction_sessions.uid, index       |
| label          | Text         | YOLO class name; index                 |
| score          | Float        | confidence 0.0–1.0; index              |
| box            | Text         | `str([x1, y1, x2, y2])` — raw string! |

**box rule:** store as `str(bbox)`, return as-is. Never call `ast.literal_eval` unless
explicitly asked. Tests assert `box == "[10, 20, 100, 200]"` (string comparison).

---

## Canonical models.py

```python
from sqlalchemy import Column, Integer, Text, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class PredictionSession(Base):
    __tablename__ = "prediction_sessions"
    uid             = Column(Text, primary_key=True)
    timestamp       = Column(DateTime, server_default=func.now(), nullable=False)
    original_image  = Column(Text, nullable=False)
    predicted_image = Column(Text, nullable=False)

class DetectionObjectModel(Base):
    __tablename__ = "detection_objects"
    id             = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(Text, ForeignKey("prediction_sessions.uid"), nullable=False, index=True)
    label          = Column(Text, nullable=False, index=True)
    score          = Column(Float, nullable=False, index=True)
    box            = Column(Text, nullable=False)
```

`DetectionObjectModel` (not `DetectionObject`) — avoids collision with the Pydantic model in app.py.

---

## Canonical db.py

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DB_BACKEND = os.environ.get("DB_BACKEND", "sqlite").lower()

if DB_BACKEND == "postgres":
    DB_USER     = os.environ.get("DB_USER", "postgres")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
    DB_HOST     = os.environ.get("DB_HOST", "localhost")
    DB_PORT     = os.environ.get("DB_PORT", "5432")
    DB_NAME     = os.environ.get("DB_NAME", "predictions")
    DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    DATABASE_URL = "sqlite:///./predictions.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

---

## app.py patterns

### Module-level table creation (NOT inside `__main__`)
```python
Base.metadata.create_all(bind=engine)
```
Place right after `app = FastAPI()`. Pytest imports the module directly — tables must
exist before any test runs.

### Every DB endpoint signature
```python
@app.get("/predictions/recent")
def get_recent(db: Session = Depends(get_db)):
    ...
```

### ORM patterns

**INSERT:**
```python
session = PredictionSession(uid=uid, original_image=orig, predicted_image=pred)
db.add(session)
db.flush()  # satisfies FK before adding detection objects
for det in detections:
    db.add(DetectionObjectModel(prediction_uid=uid, label=det.label, score=det.score, box=str(det.box)))
db.commit()
```

**SELECT one:**
```python
session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
if not session:
    raise HTTPException(status_code=404, detail="Prediction not found")
```

**SELECT N most recent:**
```python
sessions = db.query(PredictionSession).order_by(PredictionSession.timestamp.desc()).limit(10).all()
```

**JOIN by label:**
```python
rows = (
    db.query(PredictionSession.uid, PredictionSession.timestamp,
             DetectionObjectModel.id, DetectionObjectModel.label,
             DetectionObjectModel.score, DetectionObjectModel.box)
    .join(DetectionObjectModel, PredictionSession.uid == DetectionObjectModel.prediction_uid)
    .filter(DetectionObjectModel.label == label)
    .order_by(PredictionSession.timestamp, DetectionObjectModel.id)
    .all()
)
```

**SELECT by score:**
```python
objects = (
    db.query(DetectionObjectModel)
    .filter(DetectionObjectModel.score >= min_score)
    .order_by(DetectionObjectModel.id)
    .all()
)
```

---

## Test isolation — conftest.py

**Never** monkeypatch `DB_PATH` — it has no effect on SQLAlchemy. Use `dependency_overrides`:

```python
@pytest.fixture(autouse=True)
def setup_db_and_dirs(tmp_path, monkeypatch):
    import app as app_module
    test_engine = create_engine(
        f"sqlite:///{tmp_path}/test.db", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=test_engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    monkeypatch.setattr(app_module, "UPLOAD_DIR", str(tmp_path / "uploads/original"))
    monkeypatch.setattr(app_module, "PREDICTED_DIR", str(tmp_path / "uploads/predicted"))
    os.makedirs(str(tmp_path / "uploads/original"), exist_ok=True)
    os.makedirs(str(tmp_path / "uploads/predicted"), exist_ok=True)
    yield
    app.dependency_overrides.clear()
```

**Seed data via ORM — never old helper functions:**
```python
@pytest.fixture
def seeded_db():
    db: Session = next(app.dependency_overrides[get_db]())
    db.add_all([
        PredictionSession(uid="abc-123", original_image="uploads/original/abc-123.jpg",
                          predicted_image="uploads/predicted/abc-123.jpg"),
    ])
    db.flush()
    db.add_all([
        DetectionObjectModel(prediction_uid="abc-123", label="person", score=0.91, box="[10, 20, 100, 200]"),
    ])
    db.commit()
    db.close()
```

---

## Adding a column
1. Add to `models.py`
2. Delete `predictions.db` — `create_all()` rebuilds it
3. Update the writing endpoint
4. Update `seeded_db` in conftest if tests need that column

## Adding a table
1. New class inheriting `Base` in `models.py`
2. `create_all()` picks it up automatically
3. New endpoint with `Depends(get_db)`

---

## Hard rules — treat as build failures

| Rule | Detail |
|------|--------|
| No `import sqlite3` | Anywhere in app.py, db.py, models.py |
| No raw SQL strings | No SELECT/INSERT/CREATE TABLE in app.py |
| No old helpers | No `init_db()`, `save_prediction_session()`, `save_detection_object()` |
| `Depends(get_db)` | On every endpoint that reads or writes DB |
| `create_all` at module level | Not inside `__main__` |
| `box` stays a raw string | In label/score endpoints — tests assert string equality |
| API unchanged | Same paths, status codes, response shapes after any refactor |

---

## Verification — run after every change

```bash
grep -n "import sqlite3" services/yolo/app.py      # must be empty
grep -n "DB_PATH" services/yolo/app.py             # must be empty
cd services/yolo && pytest --tb=short -q           # all tests must pass
python app.py                                       # must start on port 8080
```
