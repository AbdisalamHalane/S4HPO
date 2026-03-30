import subprocess
import re
import random
import math
import ray
from ray import tune
from ray.air import session
from ray.tune.search.optuna import OptunaSearch
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

    def sample_unique_configs(self, num_configs, max_attempts=10000):
        unique_configs = []
        seen = set()
        attempts = 0

        while len(unique_configs) < num_configs and attempts < max_attempts:
            sampled_config = self.sample_config()

            config_key = tuple(
                (key, round(value, 12) if isinstance(value, float) else value)
                for key, value in sorted(sampled_config.items())
            )

            if config_key not in seen:
                seen.add(config_key)
                unique_configs.append(sampled_config)

            attempts += 1

        if len(unique_configs) < num_configs:
            raise ValueError(
                f"Could only sample {len(unique_configs)} unique configs, "
                f"but needed {num_configs}."
            )

        return unique_configs

    def run_trial_manual_successive_halving(self, config, max_epochs):
        sampled_config = config["sampled_config"]
        trial_result = self.evaluate_trial(sampled_config, max_epochs)

        metric_name = self.config_loader.get_result_metric()

        session.report({
            metric_name: trial_result[metric_name],
            "returncode": trial_result["returncode"],
            "sampled_config": sampled_config,
            "epochs": max_epochs
        })

    def sort_stage_results(self, stage_results):
        metric_name = self.config_loader.get_result_metric()
        mode = self.config_loader.get_result_mode()
        reverse_sort = (mode == "max")

        return sorted(
            stage_results,
            key=lambda result: result["metric_value"],
            reverse=reverse_sort
        )

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

        matches = re.findall(regex_pattern, output_text)

        ## check to see if we have found a metric using regex
        ## grab the metric from the latest. 
        if matches and result.returncode == 0:
            result_value = float(matches[-1])
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

    def run_successive_halving_stage(self, stage_index, stage_configs, stage_epochs):
        metric_name = self.config_loader.get_result_metric()

        trainable = tune.with_resources(
            tune.with_parameters(
                self.run_trial_manual_successive_halving,
                max_epochs=stage_epochs
            ),
            resources={
                "cpu": self.config_loader.get_cpu_num(),
                "gpu": self.config_loader.get_gpu_num()
            }
        )

        tuner = tune.Tuner(
            trainable,
            param_space={
                "sampled_config": tune.grid_search(stage_configs)
            },
            tune_config=tune.TuneConfig(
                metric=metric_name,
                mode=self.config_loader.get_result_mode()
            ),
            run_config=RunConfig(
                name=f"successive_halving_stage_{stage_index + 1}",
                storage_path=self.config_loader.get_ray_results_dir()
            )
        )

        results_grid = tuner.fit()

        stage_results = []

        for result in results_grid:
            sampled_config = result.config["sampled_config"]
            metric_value = result.metrics.get(metric_name, float("-inf"))
            returncode = result.metrics.get("returncode", 1)

            stage_results.append({
                "sampled_config": sampled_config,
                "metric_value": metric_value,
                "returncode": returncode,
                "stage_index": stage_index,
                "epochs": stage_epochs
            })

        return self.sort_stage_results(stage_results)

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
        num_trials = self.config_loader.get_stage_num_trials("random_search", stage_index)
        max_epochs = self.config_loader.get_stage_epochs("random_search", stage_index)

        trainable = tune.with_resources(
            tune.with_parameters(self.run_trial, max_epochs=max_epochs),
            resources={
                "cpu": self.config_loader.get_cpu_num(),
                "gpu": self.config_loader.get_gpu_num()
            }
        )

        storage_path = self.config_loader.get_ray_results_dir()
        exp_name = "random_search"
        exp_path = f"{storage_path}/{exp_name}"

        if tune.Tuner.can_restore(exp_path):
            print(f"Restoring Ray Tune experiment from: {exp_path}")
            tuner = tune.Tuner.restore(
                exp_path,
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=True,
                restart_errored=False,
            )
        else:
            print(f"Starting new Ray Tune experiment at: {exp_path}")
            tuner = tune.Tuner(
                trainable,
                param_space=search_space,
                tune_config=tune.TuneConfig(
                    metric=self.config_loader.get_result_metric(),
                    mode=self.config_loader.get_result_mode(),
                    num_samples=num_trials
                ),
                run_config=RunConfig(
                    name=exp_name,
                    storage_path=storage_path
                )
            )

        results = tuner.fit()
        return results

    def optimize_successive_halving(self):
        ray.shutdown()
        ray.init(ignore_reinit_error=True, log_to_driver=True)

        ## get the number of stages. 

        stages = self.config_loader.get_hpo_method_stages("successive_halving")

        ## if not stages are specified this step will early exit on error.

        initial_num_trials = stages[0]["num_trials"]
        surviving_configs = self.sample_unique_configs(initial_num_trials)

        all_stage_results = []

        for stage_index, stage in enumerate(stages):
            stage_num_trials = stage["num_trials"]
            stage_epochs = stage["epochs"]

            if len(surviving_configs) < stage_num_trials:
                raise ValueError(
                    f"Stage {stage_index + 1} requires {stage_num_trials} configs, "
                    f"but only {len(surviving_configs)} are available."
                )

            current_stage_configs = surviving_configs[:stage_num_trials]

            print(
                f"Running stage {stage_index + 1}: "
                f"{len(current_stage_configs)} configs for {stage_epochs} epoch(s)"
            )

            ranked_stage_results = self.run_successive_halving_stage(
                stage_index=stage_index,
                stage_configs=current_stage_configs,
                stage_epochs=stage_epochs
            )

            all_stage_results.append(ranked_stage_results)

            print(f"Stage {stage_index + 1} results:")
            for rank, result in enumerate(ranked_stage_results, start=1):
                print(
                    f"  Rank {rank}: "
                    f"{metric_name if (metric_name := self.config_loader.get_result_metric()) else 'metric'}="
                    f"{result['metric_value']}, "
                    f"config={result['sampled_config']}"
                )

            if stage_index < len(stages) - 1:
                next_stage_num_trials = stages[stage_index + 1]["num_trials"]
                surviving_configs = [
                    result["sampled_config"]
                    for result in ranked_stage_results[:next_stage_num_trials]
                ]
            else:
                surviving_configs = [
                    result["sampled_config"]
                    for result in ranked_stage_results
                ]

        final_results = all_stage_results[-1]
        best_result = final_results[0] if len(final_results) > 0 else None

        return {
            "best_result": best_result,
            "all_stage_results": all_stage_results
        }


    def optimize_bayesian_optimization(self):
        ray.shutdown()
        ray.init(ignore_reinit_error=True, log_to_driver=True)

        search_space = self.build_search_space()

        stage_index = 0
        num_trials = self.config_loader.get_stage_num_trials("bayesian_optimization", stage_index)
        max_epochs = self.config_loader.get_stage_epochs("bayesian_optimization", stage_index)

        ## ray tune requires the actual search algorithm (for Bayesian optimization) from an external library, in this case 
        ## we use Optuna Search. 
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

        storage_path = self.config_loader.get_ray_results_dir()
        exp_name = "bayesian_optimization"
        exp_path = f"{storage_path}/{exp_name}"

        ## checks to see if we can restore our ray tune search via the logs in the case that we 
        ## had an early exit, resume if possible other we keep going.
        if tune.Tuner.can_restore(exp_path):
            print(f"Restoring Ray Tune experiment from: {exp_path}")
            tuner = tune.Tuner.restore(
                exp_path,
                trainable=trainable,
                resume_unfinished=True,
                resume_errored=True,
                restart_errored=False,
            )
        else:
            print(f"Starting new Ray Tune experiment at: {exp_path}")
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
                    name=exp_name,
                    storage_path=storage_path
                )
            )

        ## return results. 
        results = tuner.fit()
        return results
