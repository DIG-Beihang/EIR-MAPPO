from eir_mappo.common.base_logger import BaseLogger

class CoverLogger(BaseLogger):
    def get_task_name(self):
        obs_mode = self.env_args["obs_mode"].replace("_", "")
        dynamics = self.env_args["dynamics"].replace("_", "")
        return f"{obs_mode}-{dynamics}"
        # return f"{self.env_args['scenario']}"