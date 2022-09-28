import argparse
import os

import torch
from fairseq.file_io import PathManager
from torch.serialization import default_restore_location

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, help="Path to model checkpoint")
  parser.add_argument('--output_dir', type=str, help="Output dir. Adapter will be save in the output dir with name "
                                                     "adapter_sublayers.pt")
  args = parser.parse_args()

  model_ckpt = args.model
  adapter_ckpt = f"{args.output_dir}/adapter_sublayers.pt"

  with PathManager.open(model_ckpt, "rb") as f:
    state = torch.load(
      f, map_location=lambda s, l: default_restore_location(s, "cpu")
    )

  model_state_dict = state['model']
  adapter_state_dict = {}
  for key in model_state_dict:
    if 'adapters' in key:
      adapter_state_dict[key] = model_state_dict[key]

  if len(adapter_state_dict) == 0:
    print(f"Adapter sublayers not found in model checkpoint {model_ckpt}")
  else:
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Save adapter sublayers to {adapter_ckpt}")
    torch.save(adapter_state_dict, adapter_ckpt)