
import logging

from allennlp import commands
import wandb

logger = logging.getLogger(__name__)
wandb.init(project="test-project", entity="stcl")
commands.main()
