---
name: yolo-api-data-layer
description: >
  Activate when the task touches the YOLO service database layer: adding or changing API
  endpoints that read or write data, migrating from raw SQL to SQLAlchemy, adding columns
  or tables to prediction_sessions or detection_objects, writing or fixing tests for any
  DB-touching code, or switching the database backend between SQLite and PostgreSQL.
---

# yolo-api-data-layer

> **IMPORTANT: Read this entire file before taking any action.** Do not explore the project,
> read source files, or make any changes until you have finished reading this skill. All
> necessary context (schema, ORM patterns, hard rules, test isolation) is documented here —
> start from this skill, not from the codebase.

## Activate on prompts like

- "add an endpoint GET /predictions/recent that returns the 10 most recent sessions"
- "refactor the api to use sqlalchemy"
- "add a UserFeedback table to track user ratings per prediction"
- "write tests for the /predict endpoint"
- "the database layer doesn't follow our architectural design, fix it"
- "delete a prediction session and all its detection objects by uid"
- "add a column `processing_time_ms` to the prediction_sessions table"
- "make the database backend configurable so we can use postgres in production"

---

## Project structure

```
services/yolo/
├── app.py          ← FastAPI app; endpoints use db: Session = Depends(get_db)
├── models.py       ← SQLAlchemy ORM models (PredictionSession, DetectionObjectModel)
├── db.py           ← engine factory, SessionLocal, get_db() dependency
├── requirements.txt ← must include sqlalchemy>=2.0 and psycopg2-binary>=2.9
└── tests/
    └── conftest.py ← sets up isolated test DB via dependency override
```

**Read these files before making any changes:**
1. `services/yolo/models.py` — current schema
2. `services/yolo/db.py` — engine configuration
3. `services/yolo/app.py` — endpoint signatures
4. `services/yolo/tests/conftest.py` — test isolation pattern

---

## Database schema

### prediction_sessions
| Column         | SQLAlchemy type | Notes                         |
|----------------|-----------------|-------------------------------|
| uid            | Text, PK        | UUID string                   |
| timestamp      | DateTime        | server_default=func.now()     |
| original_image | Text            | filesystem path               |
| predicted_image| Text            | filesystem path               |

### detection_objects
| Column         | SQLAlchemy type | Notes                                      |
|----------------|-----------------|--------------------------------------------|
| id             | Integer, PK     | autoincrement=True                         |
| prediction_uid | Text, FK        | references prediction_sessions.uid, index  |
| label          | Text            | YOLO class name ("person", "car", …); index|
| score          | Float           | confidence 0.0–1.0; index                  |
| box            | Text            | str([x1, y1, x2, y2]) — raw string!       |

**`box` storage rule:** always store as `str(bbox)` where `bbox` is a Python list
(e.g. `"[10, 20, 100, 200]"`). Return as-is in API responses — do NOT call
`ast.literal_eval` unless the task explicitly requires a list[float] return type.
Existing tests assert `box == "[10, 20, 100, 200]"` (string comparison).

---

## models.py — canonical implementation

```python
from sqlalchemy import Column, Integer, Text, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class PredictionSession(Base):
    __tablename__ = "prediction_sessions"

    uid = Column(Text, primary_key=True)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False)
    original_image = Column(Text, nullable=False)
    predicted_image = Column(Text, nullable=False)


class DetectionObjectModel(Base):
    __tablename__ = "detection_objects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_uid = Column(Text, ForeignKey("prediction_sessions.uid"), nullable=False, index=True)
    label = Column(Text, nullable=False, index=True)
    score = Column(Float, nullable=False, index=True)
    box = Column(Text, nullable=False)
```

The SQLAlchemy model is named `DetectionObjectModel` (not `DetectionObject`) to avoid
a name collision with the Pydantic `DetectionObject` class already in `app.py`.

---

## db.py — canonical implementation

```python
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

DB_BACKEND = os.environ.get("DB_BACKEND", "sqlite").lower()

if DB_BACKEND == "postgres":
    DB_USER = os.environ.get("DB_USER", "postgres")
    DB_PASSWORD = os.environ.get("DB_PASSWORD", "postgres")
    DB_HOST = os.environ.get("DB_HOST", "localhost")
    DB_PORT = os.environ.get("DB_PORT", "5432")
    DB_NAME = os.environ.get("DB_NAME", "predictions")
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

**Environment variables:**
- `DB_BACKEND` — `sqlite` (default) or `postgres`
- `DB_USER`, `DB_PASSWORD` — postgres credentials (never hardcode)
- `DB_HOST` — postgres host (default: `localhost`)
- `DB_PORT` — postgres port (default: `5432`)
- `DB_NAME` — postgres database name (default: `predictions`)

---

## app.py — required patterns

### Imports to add
```python
from sqlalchemy.orm import Session
from fastapi import Depends
from db import get_db, engine
from models import Base, PredictionSession, DetectionObjectModel
```

### Startup table creation (module level — NOT inside `if __name__ == "__main__"`)
```python
Base.metadata.create_all(bind=engine)
```
Place this immediately after `app = FastAPI()`. It must be at module level because pytest
imports the module directly without going through `__main__`, and the test suite expects
tables to exist on import.

### Remove entirely
- `import sqlite3`
- `DB_PATH = "predictions.db"`
- `def init_db():`
- `def save_prediction_session(...):`
- `def save_detection_object(...):`

### Endpoint signature pattern
Every endpoint that touches the DB gains `db: Session = Depends(get_db)`:
```python
@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...), db: Session = Depends(get_db)):
    ...
```

### ORM query patterns

**INSERT a session + its objects:**
```python
session = PredictionSession(uid=uid, original_image=original_path, predicted_image=predicted_path)
db.add(session)
db.flush()   # flush so FK is satisfied before adding objects

obj = DetectionObjectModel(prediction_uid=uid, label=label, score=score, box=str(bbox))
db.add(obj)

db.commit()
```

**SELECT one session by uid:**
```python
session = db.query(PredictionSession).filter(PredictionSession.uid == uid).first()
if not session:
    raise HTTPException(status_code=404, detail="Prediction not found")
```

**SELECT all objects for a session:**
```python
objects = db.query(DetectionObjectModel).filter(DetectionObjectModel.prediction_uid == uid).all()
```

**SELECT N most recent sessions:**
```python
sessions = (
    db.query(PredictionSession)
    .order_by(PredictionSession.timestamp.desc())
    .limit(n)
    .all()
)
```

**JOIN sessions + objects filtered by label:**
```python
rows = (
    db.query(
        PredictionSession.uid,
        PredictionSession.timestamp,
        DetectionObjectModel.id,
        DetectionObjectModel.label,
        DetectionObjectModel.score,
        DetectionObjectModel.box,
    )
    .join(DetectionObjectModel, PredictionSession.uid == DetectionObjectModel.prediction_uid)
    .filter(DetectionObjectModel.label == label)
    .order_by(PredictionSession.timestamp, DetectionObjectModel.id)
    .all()
)
```

**SELECT objects by score threshold:**
```python
objects = (
    db.query(DetectionObjectModel)
    .filter(DetectionObjectModel.score >= min_score)
    .order_by(DetectionObjectModel.id)
    .all()
)
```

---

## tests/conftest.py — test isolation pattern

**Do NOT** monkeypatch `DB_PATH` — it has no effect on a live SQLAlchemy engine.

Use FastAPI's dependency override mechanism instead:

```python
from app import app
from db import get_db
from models import Base, PredictionSession, DetectionObjectModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

@pytest.fixture(autouse=True)
def setup_db_and_dirs(tmp_path, monkeypatch):
    import app as app_module

    db_file = str(tmp_path / "test.db")
    test_engine = create_engine(
        f"sqlite:///{db_file}", connect_args={"check_same_thread": False}
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

    upload_dir = str(tmp_path / "uploads" / "original")
    predicted_dir = str(tmp_path / "uploads" / "predicted")
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(predicted_dir, exist_ok=True)
    monkeypatch.setattr(app_module, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(app_module, "PREDICTED_DIR", predicted_dir)

    yield
    app.dependency_overrides.clear()
```

**Seed fixture — use ORM instances, not old helper functions:**
```python
@pytest.fixture
def seeded_db():
    db: Session = next(app.dependency_overrides[get_db]())
    db.add_all([
        PredictionSession(uid="abc-123", original_image="uploads/original/abc-123.jpg",
                          predicted_image="uploads/predicted/abc-123.jpg"),
        PredictionSession(uid="def-456", original_image="uploads/original/def-456.jpg",
                          predicted_image="uploads/predicted/def-456.jpg"),
    ])
    db.flush()
    db.add_all([
        DetectionObjectModel(prediction_uid="abc-123", label="person", score=0.91, box="[10, 20, 100, 200]"),
        DetectionObjectModel(prediction_uid="abc-123", label="car",    score=0.50, box="[1, 2, 3, 4]"),
        DetectionObjectModel(prediction_uid="def-456", label="dog",    score=0.49, box="[5, 6, 7, 8]"),
    ])
    db.commit()
    db.close()
```

**For unittest.TestCase-style tests** (e.g., `test_prediction_time.py`), install the
override in `setUp` and clear it in `tearDown`:
```python
def setUp(self):
    import tempfile
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models import Base
    from db import get_db

    _, self._db_file = tempfile.mkstemp(suffix=".db")
    test_engine = create_engine(
        f"sqlite:///{self._db_file}", connect_args={"check_same_thread": False}
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
    self.client = TestClient(app)

def tearDown(self):
    app.dependency_overrides.pop(get_db, None)
    os.unlink(self._db_file)
```

**Any test file that previously imported `save_prediction_session` or `save_detection_object`
must be updated.** Replace those calls with a local helper that seeds via ORM:
```python
def _seed(uid, orig, pred, detections=()):
    db: Session = next(app.dependency_overrides[get_db]())
    db.add(PredictionSession(uid=uid, original_image=orig, predicted_image=pred))
    db.flush()
    for label, score, box in detections:
        db.add(DetectionObjectModel(prediction_uid=uid, label=label, score=score, box=str(box)))
    db.commit()
    db.close()
```

---

## Adding a new column

1. Add the column to the model in `models.py`.
2. For local dev: delete `predictions.db` — `Base.metadata.create_all()` recreates tables from scratch.
3. Update the endpoint that writes the new column.
4. Update `seeded_db` in `conftest.py` if the column is needed in existing tests.

## Adding a new table

1. Create a new class inheriting from `Base` in `models.py`.
2. `Base.metadata.create_all()` picks it up automatically on next startup.
3. Create a new endpoint in `app.py` following the `Depends(get_db)` pattern.

---

## Hard rules — these are build failures if violated

- **No `import sqlite3`** anywhere in `app.py`, `db.py`, or `models.py`
- **No raw SQL strings** (`SELECT`, `INSERT`, `UPDATE`, `CREATE TABLE`) in `app.py`
- **No `init_db()`, `save_prediction_session()`, `save_detection_object()`** functions
- **No `sqlite3.connect()`** in app code
- **`Depends(get_db)`** must appear on every endpoint that reads or writes the DB
- **`Base.metadata.create_all(bind=engine)`** must be at module level, not inside `__main__`
- **`box` stays a raw string** in label and score endpoint responses (existing tests assert string equality)

---

## Verification checklist

After any data layer change, run:

```bash
# Static: no forbidden patterns
grep -n "import sqlite3" services/yolo/app.py         # must be empty
grep -n "DB_PATH" services/yolo/app.py                 # must be empty
grep -rn "sqlite3.connect" services/yolo/app.py        # must be empty

# Tests
cd services/yolo
pytest --tb=short -q     # all tests must pass

# SQLite startup
python app.py            # must start on port 8080 without error

# PostgreSQL (requires Docker)
docker run --rm -e POSTGRES_USER=user -e POSTGRES_PASSWORD=pass \
  -e POSTGRES_DB=predictions -p 5432:5432 postgres
DB_BACKEND=postgres DB_USER=user DB_PASSWORD=pass python app.py
```
