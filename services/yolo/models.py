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
