from typing import Optional, List
from pydantic import BaseModel, model_validator


class MethodOfConsultEnum(str):
    TEL = 'TEL'
    WEB = 'WEB'
    LOC = 'LOC'


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
    operationRateType: Optional[int] = None
    methodsOfConsult: Optional[List[TMethodOfConsultData]] = None

    @model_validator(mode='before')
    def set_operation_rate_type(self, values):
        if values.get('invoicing', False):
            if values.get('hourlyRate'):
                values['operationRateType'] = 1
            elif values.get('unitPrice'):
                values['operationRateType'] = 2
        return values
