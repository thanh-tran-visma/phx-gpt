from typing import Optional, List
from pydantic import BaseModel, model_validator, field_serializer
from app.types.enum.operation import (
    MethodOfConsultEnum,
    OperationRateType,
    VatRate,
)


class TMethodOfConsultData(BaseModel):
    shortCode: MethodOfConsultEnum
    name: str


class PhxAppOperation(BaseModel):
    name: str
    description: Optional[str] = None
    duration: Optional[int] = 0
    forAppointment: bool = True
    vatRate: Optional[VatRate] = VatRate.NONE
    invoicing: bool
    hourlyRate: Optional[int] = None
    unitPrice: Optional[int] = 0
    operationRateType: Optional[OperationRateType] = (
        OperationRateType.UNIT_PRICE
    )
    methodsOfConsult: List[TMethodOfConsultData]
    uuid: str
    wizard: Optional[str] = None

    @model_validator(mode='before')
    def validate_and_convert(cls, values):
        # Convert vatRate to VatRate Enum if it's an integer
        vat_rate_value = values.get("vatRate")
        if isinstance(vat_rate_value, int):
            try:
                values["vatRate"] = VatRate(vat_rate_value)
            except ValueError:
                raise ValueError(f"Invalid vatRate value: {vat_rate_value}")
        return values

    @field_serializer("vatRate")
    def serialize_vat_rate(self, value: Optional[VatRate]) -> Optional[int]:
        if isinstance(value, VatRate):
            return value.value
        if isinstance(value, int):
            return value
        return None
