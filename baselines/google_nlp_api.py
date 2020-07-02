from comet_ml import Experiment
from utilities.general_utilities import get_args, get_bunch_config_from_json


def main() -> None:
    args = get_args()
    config = get_bunch_config_from_json(args.config)

    comet_experiment = Experiment(
        api_key=config.comet_api_key,
        project_name=config.comet_project_name,
        workspace=config.comet_workspace,
        disabled=not config.use_comet_experiments,
    )
    comet_experiment.set_name(config.comet_experiment_name)
    comet_experiment.log_parameters(config)


if __name__ == "__main__":
    main()
