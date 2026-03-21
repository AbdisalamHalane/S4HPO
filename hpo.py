import subprocess
import re
import random
import math
import ray
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler
from config_loader import ConfigLoader
from ray.train import RunConfig


class HPO:
    ## load contents from config.json
    def __init__(self, config_path):
        self.config_loader = ConfigLoader(config_path)

    ## build the hyperparameter commands to use with raytune
    def build_hyperparameter_command_list(self, sampled_config):
        hyperparameter_command_list = []

        for hyperparameter_name, value in sampled_config.items():
            option = self.config_loader.get_hyperparameter_option(hyperparameter_name)
            hyperparameter_command_list.append(f"{option}={value}")

        return hyperparameter_command_list

    ## here we build the command to run for raytune
    def build_command(self, sampled_config, max_epochs=None):
        base_command = list(self.config_loader.get_run_command())
        hyperparameter_command_list = self.build_hyperparameter_command_list(sampled_config)

        full_command = base_command + hyperparameter_command_list

        ## add the epochs option if config defined it to be fixed
        if max_epochs is not None:
            epoch_option = self.config_loader.get_epoch_option()
            full_command.append(f"{epoch_option}={max_epochs}")

        ## return the full command to use
        return full_command

    def get_working_dir(self):
        return self.config_loader.get_working_dir()

    ## here we run one trial and return the result
    def evaluate_trial(self, sampled_config, max_epochs):
        command = self.build_command(sampled_config, max_epochs=max_epochs)

        result = subprocess.run(
            command,
            cwd=self.get_working_dir(),
            capture_output=True,
            text=True
        )

        output_text = result.stdout + "\n" + result.stderr

        regex_pattern = self.config_loader.get_result_regex()
        metric_name = self.config_loader.get_result_metric()

        match = re.search(regex_pattern, output_text)

        ## check to see if we have found a metric using regex
        if match and result.returncode == 0:
            result_value = float(match.group(1))
        else:
            result_value = float("-inf")

        ## store the result for this trial
        trial_result = {
            "sampled_config": sampled_config,
            metric_name: result_value,
            "returncode": result.returncode
        }

        return trial_result

    ## this is used for random search / raytune / bayesian optimization
    def run_trial(self, sampled_config, max_epochs):
        trial_result = self.evaluate_trial(sampled_config, max_epochs)

        metric_name = self.config_loader.get_result_metric()

        session.report({
            metric_name: trial_result[metric_name],
            "returncode": trial_result["returncode"]
        })

    ## this is used for successive halving with raytune
    def run_trial_successive_halving(self, sampled_config, max_epochs):
        metric_name = self.config_loader.get_result_metric()

        ## here we report results after each epoch budget so raytune can stop bad trials early
        for current_epoch in range(1, max_epochs + 1):
            trial_result = self.evaluate_trial(sampled_config, current_epoch)

            session.report({
                metric_name: trial_result[metric_name],
                "returncode": trial_result["returncode"],
                "training_iteration": current_epoch
            })

            ## stop this trial if the subprocess failed
            if trial_result["returncode"] != 0:
                break

    ## here we manually sample one config from the search space
    def sample_config(self):
        sampled_config = {}

        hp_names = self.config_loader.get_hyperparameter_names()

        for hp_name in hp_names:
            space = self.config_loader.get_hyperparameter_space(hp_name)

            if space["type"] == "choice":
                sampled_config[hp_name] = random.choice(space["values"])
            elif space["type"] == "loguniform":
                log_min = math.log(space["min"])
                log_max = math.log(space["max"])
                sampled_config[hp_name] = math.exp(random.uniform(log_min, log_max))

        return sampled_config

    ## here we build the search space for raytune
    def build_search_space(self):
        search_space = {}

        hp_names = self.config_loader.get_hyperparameter_names()

        ## load config file defined search space
        for hp_name in hp_names:
            space = self.config_loader.get_hyperparameter_space(hp_name)

            if space["type"] == "choice":
                search_space[hp_name] = tune.choice(space["values"])
            elif space["type"] == "loguniform":
                search_space[hp_name] = tune.loguniform(space["min"], space["max"])

        return search_space

    def optimize_random_search(self):
        ray.shutdown()
        ray.init(ignore_reinit_error=True, log_to_driver=True)

        search_space = self.build_search_space()

        stage_index = 0
        ## here we load the number of trials for this search method from the config
        num_trials = self.config_loader.get_stage_num_trials("random_search", stage_index)
        ## here we load the max number of epochs for this search method
        max_epochs = self.config_loader.get_stage_epochs("random_search", stage_index)

        trainable = tune.with_resources(
            tune.with_parameters(self.run_trial, max_epochs=max_epochs),
            resources={
                "cpu": self.config_loader.get_cpu_num(),
                "gpu": self.config_loader.get_gpu_num()
            }
        )

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric=self.config_loader.get_result_metric(),
                mode=self.config_loader.get_result_mode(),
                num_samples=num_trials
            ),
            run_config=RunConfig(
                name="random_search",
                storage_path=self.config_loader.get_ray_results_dir()
            )
        )

        results = tuner.fit()
        return results

    def optimize_successive_halving(self):
        ray.shutdown()
        ray.init(ignore_reinit_error=True, log_to_driver=True)

        search_space = self.build_search_space()

        ## load the stages for successive halving from the config
        stages = self.config_loader.get_hpo_method_stages("successive_halving")

        ## here we use the first stage number of trials as the total number of sampled configs
        num_trials = stages[0]["num_trials"]

        ## here we use the last stage epoch count as the maximum training budget
        max_epochs = stages[-1]["epochs"]

        ## here we estimate the reduction factor from the first two stages
        if len(stages) >= 2:
            first_stage_trials = stages[0]["num_trials"]
            second_stage_trials = stages[1]["num_trials"]
            reduction_factor = max(2, round(first_stage_trials / second_stage_trials))
        else:
            reduction_factor = 2

        ## here we define the successive halving scheduler using raytune
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_epochs,
            grace_period=1,
            reduction_factor=reduction_factor
        )

        trainable = tune.with_resources(
            tune.with_parameters(self.run_trial_successive_halving, max_epochs=max_epochs),
            resources={
                "cpu": self.config_loader.get_cpu_num(),
                "gpu": self.config_loader.get_gpu_num()
            }
        )

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric=self.config_loader.get_result_metric(),
                mode=self.config_loader.get_result_mode(),
                num_samples=num_trials,
                scheduler=scheduler
            ),
            run_config=RunConfig(
                name="successive_halving",
                storage_path=self.config_loader.get_ray_results_dir()
            )
        )

        results = tuner.fit()
        return results
        
    def optimize_bayesian_optimization(self):
        ray.shutdown()
        ray.init(ignore_reinit_error=True, log_to_driver=True)

        search_space = self.build_search_space()

        stage_index = 0
        ## here we load the number of trials for this search method from the config
        num_trials = self.config_loader.get_stage_num_trials("bayesian_optimization", stage_index)
        ## here we load the max number of epochs for this search method
        max_epochs = self.config_loader.get_stage_epochs("bayesian_optimization", stage_index)

        ## here we define the bayesian optimization search algorithm
        search_alg = OptunaSearch(
            metric=self.config_loader.get_result_metric(),
            mode=self.config_loader.get_result_mode()
        )

        trainable = tune.with_resources(
            tune.with_parameters(self.run_trial, max_epochs=max_epochs),
            resources={
                "cpu": self.config_loader.get_cpu_num(),
                "gpu": self.config_loader.get_gpu_num()
            }
        )

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric=self.config_loader.get_result_metric(),
                mode=self.config_loader.get_result_mode(),
                num_samples=num_trials,
                search_alg=search_alg
            ),
            run_config=RunConfig(
                name="bayesian_optimization",
                storage_path=self.config_loader.get_ray_results_dir()
            )
        )

        results = tuner.fit()
        return results