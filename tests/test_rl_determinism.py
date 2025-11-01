from rl.env import TysiacEnv


def run_fixed_sequence(seed: int):
    env = TysiacEnv(seed=seed)
    obs = env.reset()
    observations = [obs.vector[:]]
    rewards = []
    dones = []
    infos = []

    for _ in range(40):
        legal = [idx for idx, flag in enumerate(obs.legal_mask) if flag]
        if not legal:
            break
        action = legal[0]
        obs, reward, done, info = env.step(action)
        observations.append(obs.vector[:])
        rewards.append(reward)
        dones.append(done)
        infos.append(info)
        if done:
            break

    return observations, rewards, dones, infos


def test_environment_determinism():
    obs1, rewards1, dones1, infos1 = run_fixed_sequence(seed=42)
    obs2, rewards2, dones2, infos2 = run_fixed_sequence(seed=42)

    assert obs1 == obs2
    assert rewards1 == rewards2
    assert dones1 == dones2
    assert infos1 == infos2
