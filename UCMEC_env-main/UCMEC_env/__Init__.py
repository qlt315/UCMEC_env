from gym.envs.registration import register

register(
    id='MA-UCMEC-v0',
    entry_point='UCMEC_env.MA_UCMEC_env_static_discrete:MA_UCMEC',
)

register(
    id='MA-UCMEC-v1',
    entry_point='UCMEC_env.MA_UCMEC_env_static_continuous:MA_UCMEC',
)

register(
    id='SA-UCMEC-v0',
    entry_point='UCMEC_env.SA_UCMEC_env_continuous:SA_UCMEC',
)
