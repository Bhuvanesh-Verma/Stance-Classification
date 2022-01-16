from typing import Dict, Any

from allennlp.training.callbacks.wandb import WandBCallback
from overrides import overrides
from allennlp.training.callbacks.callback import TrainerCallback

@TrainerCallback.register("custom_wandb")
class CustomWandBCallback(WandBCallback):
    @overrides
    def on_end(
            self,
            trainer: "GradientDescentTrainer",
            metrics: Dict[str, Any] = None,
            epoch: int = None,
            is_primary: bool = True,
            **kwargs,
    ) -> None:
        self.wandb.log(metrics)
        if is_primary:
            self.close()