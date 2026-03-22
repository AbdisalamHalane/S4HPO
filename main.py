from hpo import HPO


def print_separator():
    print("=" * 80)


def print_random_search_results(results, metric_name):
    print_separator()
    print("RANDOM SEARCH RESULTS")
    print_separator()

    best_result = results.get_best_result(metric=metric_name, mode="max")
    print(f"Best {metric_name}: {best_result.metrics.get(metric_name)}")
    print(f"All reported metrics: {best_result.metrics}")
    print()


def print_bayesian_optimization_results(results, metric_name):
    print_separator()
    print("BAYESIAN OPTIMIZATION RESULTS")
    print_separator()

    best_result = results.get_best_result(metric=metric_name, mode="max")
    print(f"Best {metric_name}: {best_result.metrics.get(metric_name)}")
    print(f"All reported metrics: {best_result.metrics}")
    print()


def print_successive_halving_results(stage_results, metric_name):
    print_separator()
    print("SUCCESSIVE HALVING RESULTS")
    print_separator()

    for stage_index, trials in enumerate(stage_results):
        print(f"Stage {stage_index + 1}:")
        for trial_index, trial_result in enumerate(trials):
            print(
                f"  Trial {trial_index + 1}: "
                f"config={trial_result['sampled_config']}, "
                f"{metric_name}={trial_result[metric_name]}, "
                f"returncode={trial_result['returncode']}"
            )
        print()

    final_stage = stage_results[-1]
    best_trial = final_stage[0]

    print("Best final-stage trial:")
    print(f"Config: {best_trial['sampled_config']}")
    print(f"{metric_name}: {best_trial[metric_name]}")
    print(f"Return code: {best_trial['returncode']}")
    print()


def main():
    config_path = "config.json"

    hpo = HPO(config_path)
    metric_name = hpo.config_loader.get_result_metric()

    #random_search_results = hpo.optimize_random_search()
    #print_random_search_results(random_search_results, metric_name)

    bayesian_optimization_results = hpo.optimize_bayesian_optimization()
    print_bayesian_optimization_results(bayesian_optimization_results, metric_name)

    #successive_halving_results = hpo.optimize_successive_halving()
    #print_successive_halving_results(successive_halving_results, metric_name)


if __name__ == "__main__":
    main()