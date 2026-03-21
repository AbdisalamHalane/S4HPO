## load config file... 
import json

class ConfigLoader:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config_data = self.load_config()

        self.experiment_name = self.config_data["experiment_name"]
        self.run_command = self.config_data["run_command"]
        self.epoch_option = self.config_data["epoch_option"]
        self.result = self.config_data["result"]
        self.hpo_methods = self.config_data["hpo_methods"]
        self.hyperparameter_commands = self.config_data["hyperparameter_commands"]
        self.paths = self.config_data["paths"]
        self.resources = self.config_data["resources"]

    def load_config(self):
        with open(self.config_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def get_experiment_name(self):
        return self.experiment_name

    def get_run_command(self):
        return self.run_command

    def get_epoch_option(self):
        return self.epoch_option

    def get_result(self):
        return self.result

    def get_result_metric(self):
        return self.result["metric"]

    def get_result_mode(self):
        return self.result["mode"]

    def get_result_regex(self):
        return self.result["regex"]

    def get_hpo_methods(self):
        return self.hpo_methods

    def get_hpo_method_names(self):
        return list(self.hpo_methods.keys())

    def get_hpo_method_stages(self, method_name):
        return self.hpo_methods[method_name]["stages"]

    def get_num_stages(self, method_name):
        return len(self.hpo_methods[method_name]["stages"])

    def get_stage_num_trials(self, method_name, stage_index):
        return self.hpo_methods[method_name]["stages"][stage_index]["num_trials"]

    def get_stage_epochs(self, method_name, stage_index):
        return self.hpo_methods[method_name]["stages"][stage_index]["epochs"]

    def get_hyperparameter_commands(self):
        return self.hyperparameter_commands

    def get_paths(self):
        return self.paths

    def get_working_dir(self):
        return self.paths["working_directory"]

    def get_hyperparameter_names(self):
        return list(self.hyperparameter_commands.keys())

    def get_hyperparameter_option(self, hyperparameter_name):
        return self.hyperparameter_commands[hyperparameter_name]["option"]

    def get_hyperparameter_space(self, hyperparameter_name):
        return self.hyperparameter_commands[hyperparameter_name]["space"]
    
    def get_resources(self):
        return self.resources

    def get_cpu_num(self):
        return self.resources["cpu_num"]

    def get_gpu_num(self):
        return self.resources["gpu_num"]
    
    def get_results_dir(self):
        return self.paths["results_dir"]

    def get_ray_results_dir(self):
        return self.paths["ray_results_dir"]
