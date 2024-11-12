from typing import Optional, List
from pydantic import BaseModel, model_validator

from app.types.enum.operation import (
    MethodOfConsultEnum,
    OperationRateType,
    VatRate,
)


class TMethodOfConsultData(BaseModel):
    shortCode: MethodOfConsultEnum
    name: str


class PhxAppOperation(BaseModel):
    name: str  # required
    description: Optional[str] = None  # optional
    duration: Optional[int] = 0  # default to 0 if not provided
    forAppointment: bool = True  # default to True
    vatRate: Optional[VatRate] = (
        VatRate.LOW.value
    )  # required, default to LOW if invoicing is true
    invoicing: bool  # optional
    hourlyRate: Optional[int] = None  # optional
    unitPrice: Optional[int] = 0  # default to 0 if not provided
    operationRateType: Optional[OperationRateType] = (
        OperationRateType.UNIT_PRICE
    )  # required if invoicing is true, default to UNIT_PRICE
    methodsOfConsult: List[TMethodOfConsultData]  # required
    uuid: str  # required, can be empty string
    wizard: Optional[str] = None  # null if not provided, can be empty string

    @model_validator(mode='before')
    def set_operation_rate_type(values: dict):
        invoicing = values.get('invoicing', False)
        hourly_rate = values.get('hourlyRate')
        unit_price = values.get('unitPrice')

        # If invoicing is enabled, set operationRateType based on available pricing details
        if invoicing:
            if hourly_rate:
                values['operationRateType'] = OperationRateType.HOURLY_RATE
            elif unit_price:
                values['operationRateType'] = OperationRateType.UNIT_PRICE
            # Ensure vatRate is provided if invoicing is true
            if 'vatRate' not in values:
                values['vatRate'] = (
                    VatRate.LOW.value
                )  # Default to LOW if not provided
        else:
            # If invoicing is disabled, unitPrice must be provided
            if unit_price is None:
                raise ValueError(
                    "Unit price is required when invoicing is false."
                )

        # Ensure methods of consultation are provided
        if not values.get('methodsOfConsult'):
            raise ValueError(
                "Methods of consultation are required (TEL, WEB, LOC)."
            )

        return values
