from pydantic import BaseModel

from app.types.enum.instruction import CRUD, InstructionList


class DecisionInstruction(BaseModel):
    instruction: InstructionList = InstructionList.DEFAULT.value
    crud: CRUD = CRUD.NONE.value
    personal_data: bool = False
