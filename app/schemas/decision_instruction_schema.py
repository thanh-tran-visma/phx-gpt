from typing import Optional

from pydantic import BaseModel

from app.types.enum.instruction import CRUD, InstructionList


class DecisionInstruction(BaseModel):
    instruction: InstructionList = InstructionList.DEFAULT
    crud: Optional[CRUD] = CRUD.NONE.value
    sensitive_data: Optional[bool] = False
