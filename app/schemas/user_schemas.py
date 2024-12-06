from pydantic import BaseModel, ConfigDict
from sqlalchemy import Column, Integer


class UserBase(BaseModel):
    id = Column(Integer, primary_key=True, index=True)

    model_config = ConfigDict(from_attributes=True)
