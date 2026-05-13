import sys
import os
import argparse
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from misc.utils import TrainingParams
from training.trainer_retrievaler import RetrievalerTrainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HOTFormerLoc learned retrievaler')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model-specific configuration file')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='HOTFormerLoc checkpoint used to initialise the encoder and pooling layer.')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from a learned retrievaler checkpoint.')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    print('Training config path: {}'.format(args.config))
    print('Model config path: {}'.format(args.model_config))
    if args.pretrained_weights is not None:
        print('Pretrained HOTFormerLoc weights: {}'.format(args.pretrained_weights))
    if args.resume_from is not None:
        print('Resuming from checkpoint path: {}'.format(args.resume_from))
    print('Debug mode: {}'.format(args.debug))
    print('Verbose mode: {}'.format(args.verbose))

    if args.resume_from is None and args.pretrained_weights is None:
        raise ValueError('--pretrained_weights is required unless --resume_from is provided.')

    params = TrainingParams(
        args.config, args.model_config, debug=args.debug, verbose=args.verbose
    )

    if args.debug:
        torch.autograd.set_detect_anomaly(True)

    trainer = RetrievalerTrainer(pretrained_weights=args.pretrained_weights)
    trainer(params, checkpoint_path=args.resume_from)
