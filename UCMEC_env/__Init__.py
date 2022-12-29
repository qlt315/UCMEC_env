from gym.envs.registration import register

register(
    id='SA_UCMEC-v0',
    entry_point='gym_foo.envs:MA_UCMEC',
)

register(
    id='MA_UCMEC-v0',
    entry_point='gym_foo.envs:SA_UCMEC',
)