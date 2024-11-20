from typing import Optional, List
from pydantic import BaseModel
from app.types.enum.operation import (
    MethodOfConsultEnum,
    OperationRateType,
    VatRate,
)


class TMethodOfConsultData(BaseModel):
    shortCode: MethodOfConsultEnum
    name: str


class PhxAppOperation(BaseModel):
    name: str = "Default Operation"  # Added default value
    description: Optional[str] = None
    duration: Optional[int] = 0
    forAppointment: bool = True
    vatRate: Optional[VatRate] = VatRate.NONE
    invoicing: bool = True  # Added default value
    hourlyRate: Optional[int] = None
    unitPrice: Optional[int] = 0
    operationRateType: Optional[OperationRateType] = OperationRateType.UNIT_PRICE
    methodsOfConsult: List[TMethodOfConsultData] = []
    uuid: str = "default-uuid"
    wizard: Optional[str] = None
