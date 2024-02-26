from harl.common.base_logger import BaseLogger

class PursuitLogger(BaseLogger):
    def get_task_name(self):
        obs_mode = self.env_args["obs_mode"].replace("_", "")
        dynamics = self.env_args["dynamics"].replace("_", "")
        evader_policy = self.env_args["evader_policy"]
        return f"{obs_mode}-{dynamics}-{evader_policy}"
        # return f"{self.env_args['scenario']}"