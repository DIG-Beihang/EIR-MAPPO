from eir_mappo.algo.mappo_advt_with_belief import MAPPOAdvtBelief


class MAPPOTraitorBelief(MAPPOAdvtBelief):
    def train(self, actor_buffer, advantages, state_type):
        return self.train_adv(actor_buffer, advantages, state_type)

    def share_param_train(self, actor_buffer, advantages, num_agents, state_type):
        return self.share_param_train_adv(actor_buffer, advantages, num_agents, state_type)
    