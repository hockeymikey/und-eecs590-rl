from enum import StrEnum


class DPAlgorithm(StrEnum):
    VALUE_ITERATION = "value_iteration"
    POLICY_ITERATION = "policy_iteration"
    Q_POLICY_ITERATION = "q_policy_iteration"


class TDPredictionAlgorithm(StrEnum):
    TD_N_FORWARD = "td_n_forward"
    TD_N_BACKWARD = "td_n_backward"
    TD_LAMBDA_FORWARD = "td_lambda_forward"
    TD_LAMBDA_BACKWARD = "td_lambda_backward"


class SarsaAlgorithm(StrEnum):
    SARSA_N_FORWARD = "sarsa_n_forward"
    SARSA_N_BACKWARD = "sarsa_n_backward"
    SARSA_LAMBDA_FORWARD = "sarsa_lambda_forward"
    SARSA_LAMBDA_BACKWARD = "sarsa_lambda_backward"