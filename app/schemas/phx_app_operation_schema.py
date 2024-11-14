from app.types.enum.operation import (
    MethodOfConsultEnum,
    OperationRateType,
    VatRate,
)
from typing import Optional, List
from pydantic import BaseModel


class TMethodOfConsultData(BaseModel):
    shortCode: MethodOfConsultEnum
    name: str


class PhxAppOperation(BaseModel):
    name: str  # required
    description: Optional[str] = None  # optional
    duration: Optional[int] = 0  # default to 0 if not provided
    forAppointment: bool = True  # default to True
    vatRate: Optional[VatRate] = (
        VatRate.LOW
    )  # default to LOW if invoicing is true
    invoicing: bool  # required
    hourlyRate: Optional[int] = None  # optional
    unitPrice: Optional[int] = 0  # default to 0 if not provided
    operationRateType: Optional[OperationRateType] = (
        OperationRateType.UNIT_PRICE
    )  # default to UNIT_PRICE
    methodsOfConsult: List[TMethodOfConsultData]  # required
    uuid: str  # required, can be empty string
    wizard: Optional[str] = None  # optional
