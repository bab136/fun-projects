from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from typing import List, Dict, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class Feedback(Base):
    __tablename__ = 'feedback'
    
    id = Column(Integer, primary_key=True)
    text = Column(String)
    component = Column(String)
    condition = Column(String)
    rating = Column(String)  # 'Bad', 'OK', 'Good'
    timestamp = Column(DateTime, default=datetime.utcnow)
    feedback_metadata = Column(JSON, nullable=True)

class FeedbackStorage:
    def __init__(self, db_path: str = "feedback/feedback.db"):
        """
        Initialize the feedback storage.
        
        Args:
            db_path (str): Path to the SQLite database
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def add_feedback(self, text: str, component: str, condition: str, 
                    rating: str, metadata: Optional[Dict] = None) -> None:
        """
        Add new feedback to the database.
        
        Args:
            text (str): Original text
            component (str): Extracted component
            condition (str): Extracted condition
            rating (str): User rating ('Bad', 'OK', 'Good')
            metadata (Dict, optional): Additional metadata
        """
        session = self.Session()
        try:
            feedback = Feedback(
                text=text,
                component=component,
                condition=condition,
                rating=rating,
                feedback_metadata=metadata
            )
            session.add(feedback)
            session.commit()
            logger.info("Feedback added successfully")
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding feedback: {str(e)}")
            raise
        finally:
            session.close()
            
    def get_feedback(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Retrieve feedback entries.
        
        Args:
            limit (int, optional): Maximum number of entries to return
            
        Returns:
            List[Dict]: List of feedback entries
        """
        session = self.Session()
        try:
            query = session.query(Feedback)
            if limit:
                query = query.limit(limit)
            feedbacks = query.all()
            
            return [{
                'id': f.id,
                'text': f.text,
                'component': f.component,
                'condition': f.condition,
                'rating': f.rating,
                'timestamp': f.timestamp.isoformat(),
                'metadata': f.feedback_metadata
            } for f in feedbacks]
        finally:
            session.close()
            
    def get_feedback_by_rating(self, rating: str) -> List[Dict]:
        """
        Get feedback entries filtered by rating.
        
        Args:
            rating (str): Rating to filter by ('Bad', 'OK', 'Good')
            
        Returns:
            List[Dict]: List of feedback entries
        """
        session = self.Session()
        try:
            feedbacks = session.query(Feedback).filter(Feedback.rating == rating).all()
            return [{
                'id': f.id,
                'text': f.text,
                'component': f.component,
                'condition': f.condition,
                'rating': f.rating,
                'timestamp': f.timestamp.isoformat(),
                'metadata': f.feedback_metadata
            } for f in feedbacks]
        finally:
            session.close()
            
    def get_feedback_stats(self) -> Dict:
        """
        Get statistics about feedback ratings.
        
        Returns:
            Dict: Statistics about feedback ratings
        """
        session = self.Session()
        try:
            total = session.query(Feedback).count()
            ratings = {
                'Bad': session.query(Feedback).filter(Feedback.rating == 'Bad').count(),
                'OK': session.query(Feedback).filter(Feedback.rating == 'OK').count(),
                'Good': session.query(Feedback).filter(Feedback.rating == 'Good').count()
            }
            
            return {
                'total': total,
                'ratings': ratings,
                'percentages': {
                    k: (v / total * 100) if total > 0 else 0
                    for k, v in ratings.items()
                }
            }
        finally:
            session.close() 