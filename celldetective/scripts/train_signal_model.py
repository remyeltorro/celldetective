"""
Copright © 2023 Laboratoire Adhesion et Inflammation, Authored by Remy Torro.
"""

import argparse
import os
import json
from pathlib import Path, PurePath
from glob import glob
from tqdm import tqdm
import numpy as np
import gc
from art import tprint
from celldetective.signals import SignalDetectionModel

tprint("Train")

parser = argparse.ArgumentParser(description="Train a signal model from instructions.",
								formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-c',"--config", required=True,help="Training instructions")

args = parser.parse_args()
process_arguments = vars(args)
instructions = str(process_arguments['config'])

if os.path.exists(instructions):
	with open(instructions, 'r') as f:
		threshold_instructions = json.load(f)
		threshold_instructions.update({'n_channels': len(threshold_instructions['channel_option'])})
		if threshold_instructions['augmentation_factor']>1.0:
			threshold_instructions.update({'augment': True})
		else:
			threshold_instructions.update({'augment': False})
		threshold_instructions.update({'test_split': 0.})
else:
	print('The configuration path is not valid. Abort.')
	os.abort()


model_params = {k:threshold_instructions[k] for k in ('pretrained', 'model_signal_length', 'channel_option', 'n_channels') if k in threshold_instructions}
train_params = {k:threshold_instructions[k] for k in ('model_name', 'target_directory', 'channel_option','recompile_pretrained', 'test_split', 'augment', 'epochs', 'learning_rate', 'batch_size', 'validation_split') if k in threshold_instructions}

print(f'model params {model_params}')
print(f'train params {train_params}')

model = SignalDetectionModel(**model_params)
model.fit_from_directory(threshold_instructions['ds'], **train_params)

print('Done.')