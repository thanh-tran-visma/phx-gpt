import logging
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


class HelperFunctions:
    def __init__(self, db: Session) -> None:
        self.db: Session = db

    def commit_and_log(self, success_message: str) -> bool:
        """Commit the transaction and log a success message if successful."""
        try:
            self.db.commit()
            logger.info(success_message)
            return True
        except (IntegrityError, SQLAlchemyError) as e:
            logger.error(f"Commit error: {str(e)}", exc_info=True)
            self.db.rollback()
            return False
