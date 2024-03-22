from gym.envs.registration import register

register(
    id='MA-UCMEC-Static-v0',
    entry_point='UCMEC.envs:MA_UCMEC_Static',
)

register(
    id='MA-UCMEC-Varying-v0',
    entry_point='UCMEC.envs:MA_UCMEC_Varying',
)

register(
    id='SA-UCMEC-v0',
    entry_point='UCMEC.envs:SA_UCMEC',
)

register(
    id='MA-UCMEC-stat-noncoop-v0',
    entry_point='UCMEC.envs:MA_UCMEC_stat_noncoop',
)

register(
    id='MA-UCMEC-stat-coop-v0',
    entry_point='UCMEC.envs:MA_UCMEC_stat_coop',
)

register(
    id='MA-UCMEC-dyna-noncoop-v0',
    entry_point='UCMEC.envs:MA_UCMEC_dyna_coop',
)

register(
    id='MA-UCMEC-dyna-coop-v0',
    entry_point='UCMEC.envs:MA_UCMEC_dyna_noncoop',
)

register(
    id='MA-CBO-stat-noncoop-v0',
    entry_point='UCMEC.envs:MA_CBO_stat_noncoop',
)

register(
    id='MA-CBO-stat-coop-v0',
    entry_point='UCMEC.envs:MA_CBO_stat_coop',
)

register(
    id='MA-CBO-dyna-noncoop-v0',
    entry_point='UCMEC.envs:MA_CBO_dyna_noncoop',
)

register(
    id='MA-CBO-dyna-coop-v0',
    entry_point='UCMEC.envs:MA_CBO_dyna_coop',
)

register(
    id='MA-MPO-stat-noncoop-v0',
    entry_point='UCMEC.envs:MA_MPO_stat_noncoop',
)

register(
    id='MA-MPO-stat-coop-v0',
    entry_point='UCMEC.envs:MA_MPO_stat_coop',
)

register(
    id='MA-MPO-dyna-noncoop-v0',
    entry_point='UCMEC.envs:MA_MPO_dyna_noncoop',
)

register(
    id='MA-MPO-dyna-coop-v0',
    entry_point='UCMEC.envs:MA_MPO_dyna_coop',
)