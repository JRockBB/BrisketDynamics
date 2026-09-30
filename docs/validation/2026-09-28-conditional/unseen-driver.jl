# Prespecified after the implementation freeze, evaluated once without tuning.
const unseen_cases=[
    (suite=:core,seed=23011,times=[0.,80.,360.],channels=[[:core] for _ in 1:3]),
    (suite=:two_core,seed=23012,times=[0.,110.,390.],channels=[[:core,:second],[:second],[:core,:second]])
]
FinalConditionalValidation.validate_deterministic(output=joinpath(@__DIR__,"unseen_sparse_cooks.json"),cases=unseen_cases,nx=5)