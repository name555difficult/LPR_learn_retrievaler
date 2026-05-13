import os
import pathlib
import typing as tp

import numpy as np
import torch
import tqdm
import wandb
import submitit
from timm.optim.lamb import Lamb

from datasets.dataset_utils import make_dataloaders
from eval.pnv_evaluate_retrievaler import (
    evaluate,
    print_eval_stats_retrievaler,
    pnv_write_eval_stats_retrievaler,
)
from misc.torch_utils import to_device
from misc.utils import TrainingParams, get_datetime, set_seed
from models.losses.retrievaler_loss import RetrievalerLoss
from models.model_factory import model_factory
from models.retrievaler import build_retrievaler, load_retrievaler_config


class RetrievalerTrainer:
    """
    Trainer for learned retrievaler experiments.

    HOTFormerLoc backbone parameters are frozen. Pooling/aggregation and the
    retrievaler are optimised with retrievaler-only losses.
    """

    def __init__(self, pretrained_weights: str = None):
        self.model = None
        self.retrievaler = None
        self.optimizer = None
        self.scheduler = None
        self.params = None
        self.retrievaler_cfg = None
        self.model_pathname = None
        self.device = None
        self.wandb_id = None
        self.resume = False
        self.start_epoch = 1
        self.curr_epoch = 1
        self.best_avg_AR_1 = 0.0
        self.checkpoint_extension = '_latest.ckpt'
        self.pretrained_weights = pretrained_weights

    def __call__(
        self,
        params: TrainingParams = None,
        *args,
        **kwargs,
    ):
        checkpoint_path = kwargs.get('checkpoint_path')
        self.resume = checkpoint_path is not None
        self.params = params
        self.retrievaler_cfg = load_retrievaler_config(self.params.params_path)
        self.params.print()
        print('Retrievaler parameters:')
        for key, value in vars(self.retrievaler_cfg).items():
            print('{}: {}'.format(key, value))
        print('')

        set_seed()
        self.init_model_retrievaler_optim_sched()

        if self.resume:
            self.load_checkpoint(checkpoint_path)
        else:
            self.load_pretrained_model(self.pretrained_weights)
            s = get_datetime()
            model_name = self.params.model_params.model + '_retrievaler_' + s
            if 'SLURM_JOB_ID' in os.environ:
                model_name += f"_job{os.environ['SLURM_JOB_ID']}"
            weights_path = self.create_weights_folder(self.params.dataset_name)
            self.model_pathname = os.path.join(weights_path, model_name)
            print('Model name: {}'.format(model_name))

        self.freeze_backbone()
        self.print_info()
        self.do_train()

    def checkpoint(self, *args: tp.Any, **kwargs: tp.Any) -> submitit.helpers.DelayedSubmission:
        checkpoint_path = self.model_pathname + self.checkpoint_extension
        print(f'Training interrupted at epoch {self.curr_epoch}. '
              f'Saving ckpt to {checkpoint_path} and resubmitting.')
        if not os.path.exists(checkpoint_path):
            self.save_checkpoint(checkpoint_path)
        training_callable = RetrievalerTrainer(pretrained_weights=self.pretrained_weights)
        delayed_submission = submitit.helpers.DelayedSubmission(
            training_callable,
            self.params,
            checkpoint_path=checkpoint_path,
        )
        return delayed_submission

    def init_model_retrievaler_optim_sched(self):
        self.model = model_factory(self.params.model_params)
        self.retrievaler = build_retrievaler(
            descriptor_dim=self.params.model_params.output_dim,
            cfg=self.retrievaler_cfg,
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.retrievaler.to(self.device)
        print('Model device: {}'.format(self.device))

        self.freeze_backbone()

        if self.params.optimizer == 'Adam':
            optimizer_fn = torch.optim.Adam
        elif self.params.optimizer == 'AdamW':
            optimizer_fn = torch.optim.AdamW
        elif self.params.optimizer == 'Lamb':
            optimizer_fn = Lamb
        else:
            raise NotImplementedError(f"Unsupported optimizer: {self.params.optimizer}")

        trainable_params = [
            p for p in list(self.model.parameters()) + list(self.retrievaler.parameters())
            if p.requires_grad
        ]
        if self.params.weight_decay is None or self.params.weight_decay == 0:
            self.optimizer = optimizer_fn(trainable_params, lr=self.params.lr)
        else:
            self.optimizer = optimizer_fn(
                trainable_params, lr=self.params.lr, weight_decay=self.params.weight_decay
            )

        if self.params.scheduler is not None:
            if self.params.scheduler == 'CosineAnnealingLR':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.params.epochs + 1,
                    eta_min=self.params.min_lr
                )
            elif self.params.scheduler == 'MultiStepLR':
                self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, self.params.scheduler_milestones,
                    gamma=self.params.gamma
                )
            elif self.params.scheduler == 'ExponentialLR':
                self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.params.gamma
                )
            else:
                raise NotImplementedError(
                    'Unsupported LR scheduler: {}'.format(self.params.scheduler)
                )

        if self.params.warmup_epochs is not None:
            warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda=self.warmup
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, [warmup_scheduler, self.scheduler],
                [self.params.warmup_epochs]
            )

    def warmup(self, epoch: int):
        min_factor = 1e-3
        return max(float(epoch / self.params.warmup_epochs), min_factor)

    def freeze_backbone(self):
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        for param in self.model.pooling.parameters():
            param.requires_grad = True
        for param in self.retrievaler.parameters():
            param.requires_grad = True

    def load_pretrained_model(self, weights_path: str):
        assert weights_path is not None
        assert os.path.exists(weights_path), 'Cannot open weights: {}'.format(weights_path)
        print('Loading pretrained HOTFormerLoc weights: {}'.format(weights_path))
        state = torch.load(weights_path, map_location=self.device)
        if os.path.splitext(weights_path)[1] == '.ckpt':
            state_dict = state['model_state_dict']
        else:
            state_dict = state
        self.model.load_state_dict(state_dict)

    def load_checkpoint(self, checkpoint_path: str):
        assert os.path.exists(checkpoint_path), 'Cannot open checkpoint: {}'.format(checkpoint_path)
        self.model_pathname = checkpoint_path.split(self.checkpoint_extension)[0]
        state = torch.load(checkpoint_path, map_location=self.device)
        self.start_epoch = state['epoch']
        self.curr_epoch = self.start_epoch
        self.wandb_id = state.get('wandb_id')
        self.best_avg_AR_1 = state.get('best_avg_AR_1', 0.0)
        self.model.load_state_dict(state['model_state_dict'])
        self.retrievaler.load_state_dict(state['retrievaler_state_dict'])
        self.optimizer.load_state_dict(state['optim_state_dict'])
        if self.scheduler is not None and 'sched_state_dict' in state:
            self.scheduler.load_state_dict(state['sched_state_dict'])
        print(f'Resuming retrievaler training of {self.model_pathname} from epoch {self.start_epoch}')

    def save_checkpoint(self, checkpoint_path: str):
        print(f"[INFO] Saving checkpoint to {checkpoint_path}", flush=True)
        state = {
            'epoch': self.curr_epoch,
            'wandb_id': self.wandb_id,
            'best_avg_AR_1': self.best_avg_AR_1,
            'model_state_dict': self.model.state_dict(),
            'retrievaler_state_dict': self.retrievaler.state_dict(),
            'optim_state_dict': self.optimizer.state_dict(),
            'retrievaler_cfg': vars(self.retrievaler_cfg),
        }
        if self.scheduler is not None:
            state['sched_state_dict'] = self.scheduler.state_dict()
        torch.save(state, checkpoint_path)

    def create_weights_folder(self, dataset_name: str):
        this_file_path = pathlib.Path(__file__).parent.absolute()
        root_path, _ = os.path.split(this_file_path)
        weights_path = os.path.join(root_path, 'weights', dataset_name, 'retrievaler')
        os.makedirs(weights_path, exist_ok=True)
        assert os.path.exists(weights_path), 'Cannot create weights folder: {}'.format(weights_path)
        return weights_path

    def print_info(self):
        if hasattr(self.model, 'print_info'):
            self.model.print_info()
        model_params = sum(p.nelement() for p in self.model.parameters())
        retrievaler_params = sum(p.nelement() for p in self.retrievaler.parameters())
        trainable_model_params = sum(
            p.nelement() for p in self.model.parameters() if p.requires_grad
        )
        print(f'HOTFormerLoc parameters: {model_params}')
        print(f'Trainable HOTFormerLoc parameters: {trainable_model_params}')
        print(f'Retrievaler parameters: {retrievaler_params}')

    def prepare_phase(self, phase: str):
        if phase == 'train':
            self.model.train()
            self.model.backbone.eval()
            self.retrievaler.train()
        else:
            self.model.eval()
            self.retrievaler.eval()

    def compute_embeddings(self, batch, phase: str):
        embeddings_l = []
        batches = batch if isinstance(batch, list) else [batch]
        for minibatch in batches:
            minibatch = to_device(
                minibatch, self.device, non_blocking=True,
                construct_octree_neigh=True
            )
            y = self.model(minibatch)
            embeddings_l.append(y['global'])
        return torch.cat(embeddings_l, dim=0)

    def training_step(self, global_iter, phase, loss_fn):
        assert phase in ['train', 'val']
        batch, positives_mask, negatives_mask = next(global_iter)
        self.prepare_phase(phase)

        if phase == 'train':
            self.optimizer.zero_grad()

        with torch.set_grad_enabled(phase == 'train'):
            embeddings = self.compute_embeddings(batch, phase)
            loss, stats = loss_fn(
                self.retrievaler, embeddings, positives_mask, negatives_mask
            )
            if phase == 'train':
                loss.backward()
                self.optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stats

    def print_stats(self, phase, stats):
        s = (
            f"{phase}  loss: {stats['loss']:.4f}  "
            f"ce: {stats['ce_loss']:.4f}  "
            f"db: {stats['db_margin_loss']:.4f}  "
            f"rank: {stats['rank_loss']:.4f}  "
            f"pos_acc: {stats['positive_chunk_acc']:.3f}  "
            f"empty_acc: {stats['empty_chunk_acc']:.3f}  "
            f"chunks: {stats['num_chunks']:.0f}"
        )
        print(s, flush=True)

    def log_eval_stats(self, stats):
        eval_stats = {}
        for database_name in stats:
            eval_stats[database_name] = {
                'recall@1%': stats[database_name]['ave_one_percent_recall'],
                'recall@1': stats[database_name]['ave_recall'][0],
                'MRR': stats[database_name]['ave_mrr'],
                'survival@stage1': stats[database_name]['survival_stage1'],
                'empty_false_accept_rate': stats[database_name]['empty_false_accept_rate'],
                'positive_reject_rate': stats[database_name]['positive_reject_rate'],
            }
        return eval_stats

    def do_train(self):
        dataloaders = make_dataloaders(self.params, validation=self.params.validation)
        loss_fn = RetrievalerLoss(self.retrievaler_cfg)

        phases = ['train']
        if 'val' in dataloaders:
            phases.append('val')

        params_dict = {
            e: self.params.__dict__[e]
            for e in self.params.__dict__
            if e != 'model_params'
        }
        model_params_dict = {
            'model_params.' + e: self.params.model_params.__dict__[e]
            for e in self.params.model_params.__dict__
        }
        retrievaler_params_dict = {
            'retrievaler.' + e: v for e, v in vars(self.retrievaler_cfg).items()
        }
        params_dict.update(model_params_dict)
        params_dict.update(retrievaler_params_dict)
        if self.params.wandb and not self.params.debug:
            wandb.init(
                project='HOTFormerLoc-Retrievaler',
                config=params_dict,
                id=self.wandb_id,
                resume='allow',
            )
            self.wandb_id = wandb.run.id

        for epoch in tqdm.tqdm(
            range(self.start_epoch, self.params.epochs + 1),
            initial=self.start_epoch - 1,
            total=self.params.epochs,
        ):
            metrics = {'train': {}, 'val': {}, 'test': {}}
            for phase in phases:
                running_stats = []
                global_iter = iter(dataloaders[phase])
                count_batches = 0
                while True:
                    count_batches += 1
                    if self.params.debug and count_batches > 2:
                        break
                    try:
                        batch_stats = self.training_step(global_iter, phase, loss_fn)
                    except StopIteration:
                        break
                    running_stats.append(batch_stats)

                if len(running_stats) == 0:
                    continue

                epoch_stats = {}
                for key in running_stats[0]:
                    values = [e[key] for e in running_stats]
                    epoch_stats[key] = np.mean(values)

                self.print_stats(phase, epoch_stats)
                metrics[phase].update(epoch_stats)

            self.curr_epoch += 1
            if self.scheduler is not None:
                self.scheduler.step()

            if not self.params.debug:
                checkpoint_path = self.model_pathname + self.checkpoint_extension
                self.save_checkpoint(checkpoint_path)
                if self.params.save_freq > 0 and epoch % self.params.save_freq == 0:
                    epoch_pathname = f"{self.model_pathname}_e{epoch}.ckpt"
                    self.save_checkpoint(epoch_pathname)

            if self.params.eval_freq > 0 and epoch % self.params.eval_freq == 0:
                eval_stats = evaluate(
                    self.model, self.retrievaler, self.device, self.params,
                    self.retrievaler_cfg, log=False,
                    model_name=os.path.basename(self.model_pathname),
                    show_progress=self.params.verbose,
                )
                print_eval_stats_retrievaler(eval_stats)
                metrics['test'] = self.log_eval_stats(eval_stats)
                avg_AR_1 = metrics['test']['average']['recall@1']
                if avg_AR_1 > self.best_avg_AR_1:
                    print(
                        f"New best avg AR@1 at Epoch {epoch}: "
                        f"{self.best_avg_AR_1:.2f} -> {avg_AR_1:.2f}"
                    )
                    self.best_avg_AR_1 = avg_AR_1
                    if not self.params.debug:
                        best_model_pathname = f"{self.model_pathname}_best.ckpt"
                        self.save_checkpoint(best_model_pathname)

            if self.params.wandb and not self.params.debug:
                wandb.log(metrics)

        if not self.params.debug:
            final_model_path = self.model_pathname + '_final.ckpt'
            self.save_checkpoint(final_model_path)
            stats = evaluate(
                self.model, self.retrievaler, self.device, self.params,
                self.retrievaler_cfg, log=False,
                model_name=os.path.basename(final_model_path),
                show_progress=self.params.verbose,
            )
            print_eval_stats_retrievaler(stats)

            model_params_name = os.path.split(self.params.model_params.model_params_path)[1]
            config_name = os.path.split(self.params.params_path)[1]
            model_name = os.path.splitext(os.path.split(final_model_path)[1])[0]
            prefix = "{}, {}, {}".format(model_params_name, config_name, model_name)
            pnv_write_eval_stats_retrievaler(
                f"pnv_{self.params.dataset_name}_retrievaler_results.txt",
                prefix,
                stats,
            )

        return 1 - self.best_avg_AR_1 / 100.0
