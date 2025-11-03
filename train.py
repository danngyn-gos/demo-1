from training_task.multiclass_training_task import TrainingMultiTask
from models.sentiment_model import SentimentModel
from configs.utils import get_config
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)

args = parser.parse_args()

config = get_config(args.config_file)

model = SentimentModel(config.MODEL)

task = TrainingMultiTask(config, model)
task.start()
# task.get_predictions()
