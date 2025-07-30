import openpi.models.pi0 as pi0
import openpi.training.weight_loaders as weight_loaders


def get_deerbaby_configs():
    from openpi.training.config import (
        AssetsConfig,
        DataConfig,
        LeRobotDeerbabyDataConfig,
        TrainConfig,
    )

    return [
        #
        # Fine-tuning Deerbaby configs.
        #
        TrainConfig(
            name="pi0_move_pikachu",
            model=pi0.Pi0Config(),
            data=LeRobotDeerbabyDataConfig(
                repo_id="silverlife/move_pikachu",
                base_config=DataConfig(prompt_from_task=True),
                default_prompt="pick up the yellow pikachu and place it in the red bucket",
                use_delta_joint_actions=True,
                adapt_to_pi=True,
                assets=AssetsConfig(
                    # assets_dir="",
                    asset_id="deerbaby", # default to repo_id,
                ),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000,
            # The freeze filter defines which parameters should be frozen during training.
            # We have a convenience function in the model config that returns the default freeze filter
            # for the given model config for LoRA finetuning. Just make sure it matches the model config
            # you chose above.
            freeze_filter=pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            # Turn off EMA for LoRA finetuning.
            ema_decay=None,
        ),

        TrainConfig(
            name="pi0_open_door_office",
            model=pi0.Pi0Config(),
            data=LeRobotDeerbabyDataConfig(
                repo_id="silverlife/open_door_office",
                base_config=DataConfig(prompt_from_task=True),
                use_delta_joint_actions=True,
                adapt_to_pi=True,
                assets=AssetsConfig(asset_id="deerbaby"),
            ),
            weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
            num_train_steps=30_000 + 1,
            freeze_filter=pi0.Pi0Config(
                paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
            ).get_freeze_filter(),
            ema_decay=None,
            save_interval=5000,
        ),
    ]
