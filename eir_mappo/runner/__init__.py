from eir_mappo.runner.on_policy_ma_runner_advt_with_belief import OnPolicyMARunnerAdvtBelief

RUNNER_REGISTRY = {
    "mappo_advt_belief": OnPolicyMARunnerAdvtBelief,
    "mappo_traitor_belief": OnPolicyMARunnerAdvtBelief,
}
