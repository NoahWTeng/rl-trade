from finrl.agents.elegantrl.models import DRLAgent as DRLAgent_erl
from finrl.agents.rllib.models import DRLAgent as DRLAgent_rllib
from finrl.agents.stablebaselines3.models import DRLAgent as DRLAgent_sb3
from finrl.meta.data_processor import DataProcessor
import numpy as np
import pandas as pd


def train(
    dataset: pd.DataFrame,
    dataset_normalized: pd.DataFrame,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs
):
    dataset.drop(columns=["unix"], inplace=True)

    date = dataset["date"]

    price_array = dataset.iloc[:, 0:6]
    tech_array = dataset.iloc[:, 6:]

    data_config = {
        "price_array": price_array,
        "tech_array": tech_array,
    }

    # build environment using processed data
    env_instance = env(config=data_config)
    print(env_instance)
    # # read parameters and load agents
    # current_working_dir = kwargs.get("current_working_dir", "./" + str(model_name))

    # if drl_lib == "elegantrl":
    #     break_step = kwargs.get("break_step", 1e6)
    #     erl_params = kwargs.get("erl_params")

    #     agent = DRLAgent_erl(
    #         env=env,
    #         price_array=price_array,
    #         tech_array=tech_array,
    #         turbulence_array=turbulence_array,
    #     )

    #     model = agent.get_model(model_name, model_kwargs=erl_params)

    #     trained_model = agent.train_model(
    #         model=model, cwd=current_working_dir, total_timesteps=break_step
    #     )

    # elif drl_lib == "rllib":
    #     total_episodes = kwargs.get("total_episodes", 100)
    #     rllib_params = kwargs.get("rllib_params")

    #     agent_rllib = DRLAgent_rllib(
    #         env=env,
    #         price_array=price_array,
    #         tech_array=tech_array,
    #         turbulence_array=turbulence_array,
    #     )

    #     model, model_config = agent_rllib.get_model(model_name)

    #     model_config["lr"] = rllib_params["lr"]
    #     model_config["train_batch_size"] = rllib_params["train_batch_size"]
    #     model_config["gamma"] = rllib_params["gamma"]

    #     trained_model = agent_rllib.train_model(
    #         model=model,
    #         model_name=model_name,
    #         model_config=model_config,
    #         total_episodes=total_episodes,
    #     )
    #     trained_model.save(current_working_dir)

    # elif drl_lib == "stable_baselines3":
    #     total_timesteps = kwargs.get("total_timesteps", 1e6)
    #     agent_params = kwargs.get("agent_params")

    #     agent = DRLAgent_sb3(env=env_instance)

    #     model = agent.get_model(model_name, model_kwargs=agent_params)
    #     trained_model = agent.train_model(
    #         model=model, tb_log_name=model_name, total_timesteps=total_timesteps
    #     )
    #     print("Training finished!")
    #     trained_model.save(current_working_dir)
    #     print("Trained model saved in " + str(current_working_dir))
    # else:
    #     raise ValueError("DRL library input is NOT supported. Please check.")


def test(
    start_date,
    end_date,
    ticker_list,
    data_source,
    time_interval,
    technical_indicator_list,
    drl_lib,
    env,
    model_name,
    if_vix=True,
    **kwargs
):

    # process data using unified data processor
    DP = DataProcessor(data_source, start_date, end_date, time_interval, **kwargs)
    price_array, tech_array, turbulence_array = DP.run(
        ticker_list, technical_indicator_list, if_vix, cache=True
    )

    np.save("./price_array.npy", price_array)
    data_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
    }
    # build environment using processed data
    env_instance = env(config=data_config)

    env_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
        "if_train": False,
    }
    env_instance = env(config=env_config)

    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    current_working_dir = kwargs.get("current_working_dir", "./" + str(model_name))
    print("price_array: ", len(price_array))

    if drl_lib == "elegantrl":
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=current_working_dir,
            net_dimension=net_dimension,
            environment=env_instance,
        )

        return episode_total_assets

    elif drl_lib == "rllib":
        # load agent
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            price_array=price_array,
            tech_array=tech_array,
            turbulence_array=turbulence_array,
            agent_path=current_working_dir,
        )

        return episode_total_assets

    elif drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env_instance, cwd=current_working_dir
        )

        return episode_total_assets
    else:
        raise ValueError("DRL library input is NOT supported. Please check.")
