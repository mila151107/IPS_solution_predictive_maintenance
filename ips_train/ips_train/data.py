import os

UDI_COL = "UDI"
PROCESS_TEMPERATURE_COL = "Process temperature [K]"
AIR_TEMPERATURE_COL = "Air temperature [K]"
TORQUE_RPM_RATIO_COL = "Torque/RPM ratio"
PRODUCT_ID_COL = "Product ID"
TYPE_COL = "Type"
FAILURE_TYPE_COL = "Failure Type"
ROTATIONAL_SPEED_COL = "Rotational speed [rpm]"
TORQUE_COL = "Torque [Nm]"
TOOL_WEAR_COL = "Tool wear [min]"
ANGULAR_SPEED_COL = "Angular speed [rad/s]"
POWER_COL = "Power [kW]"
DELTA_TEMPERATURE_COL = "Delta Temperature [K]"
STRESS_INDEX_COL = "Stress Index"
TARGET_COL = "Target"

COLUMNS_TO_DROP = [UDI_COL, PROCESS_TEMPERATURE_COL, AIR_TEMPERATURE_COL, TORQUE_RPM_RATIO_COL]
CATEGORICAL_COLS = [PRODUCT_ID_COL, TYPE_COL, FAILURE_TYPE_COL]
NUMERICAL_COLS = [
    ROTATIONAL_SPEED_COL,
    TORQUE_COL,
    TOOL_WEAR_COL,
    ANGULAR_SPEED_COL,
    POWER_COL,
    DELTA_TEMPERATURE_COL,
    STRESS_INDEX_COL,
]

INPUT_DATASET = os.path.join(__file__, "..", "..", "predictive_maintenance.csv")
PREPROCESSOR_PATH = os.path.join(__file__, "..", "..", "artifacts", "preprocessor.joblib")
