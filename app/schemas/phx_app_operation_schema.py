from typing import Optional, List
from pydantic import BaseModel, model_validator

from app.types.enum.operation import MethodOfConsultEnum, OperationRateType


class TMethodOfConsultData(BaseModel):
    shortCode: MethodOfConsultEnum
    name: str


class PhxAppOperation(BaseModel):
    name: str
    invoice_description: Optional[str] = None
    duration: Optional[int] = None
    wizard: Optional[None] = None
    invoicing: bool
    hourlyRate: Optional[int] = None
    unitPrice: Optional[int] = None
    operationRateType: Optional[OperationRateType] = None
    methodsOfConsult: Optional[List[TMethodOfConsultData]] = None

    @model_validator(mode='before')
    def set_operation_rate_type(cls, values: dict):
        invoicing = values.get('invoicing', False)
        hourly_rate = values.get('hourlyRate')
        unit_price = values.get('unitPrice')

        if invoicing:
            if hourly_rate:
                values['operationRateType'] = OperationRateType.HOURLY_RATE
            elif unit_price:
                values['operationRateType'] = OperationRateType.UNIT_PRICE
        return values
