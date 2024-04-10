import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def define_lr_scheduler(args, optimizer):
    """
    Define the learning rate scheduler
    """
    if args.train_mode in {"supervised"}:
        classifier_config = args.dataset_config[args.model]
        optimizer_config = classifier_config["optimizer"]
        scheduler_config = classifier_config["lr_scheduler"]
    elif args.stage == "pretrain":
        optimizer_config = args.dataset_config[args.learn_framework]["pretrain_optimizer"]
        scheduler_config = args.dataset_config[args.learn_framework]["pretrain_lr_scheduler"]
    elif args.stage == "finetune":
        optimizer_config = args.dataset_config[args.learn_framework]["finetune_optimizer"]
        scheduler_config = args.dataset_config[args.learn_framework]["finetune_lr_scheduler"]
    else:
        raise Exception(f"Mode: {args.mode} and stage: {args.stage} not defined.")

    if scheduler_config["name"] == "cosine":
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=(scheduler_config["train_epochs"] - scheduler_config["warmup_epochs"])
            if scheduler_config["warmup_prefix"]
            else scheduler_config["train_epochs"],
            cycle_mul=1.0,
            lr_min=optimizer_config["min_lr"],
            warmup_lr_init=optimizer_config["warmup_lr"],
            warmup_t=scheduler_config["warmup_epochs"],
            cycle_limit=1,
            t_in_epochs=True,
            warmup_prefix=scheduler_config["warmup_prefix"],
        )
    elif scheduler_config["name"] == "linear":
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=scheduler_config["train_epochs"],
            lr_min_rate=0.01,
            warmup_lr_init=optimizer_config["warmup_lr"],
            warmup_t=scheduler_config["warmup_epochs"],
            t_in_epochs=True,
        )
    elif scheduler_config["name"] == "step":
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=scheduler_config["decay_epochs"],
            decay_rate=scheduler_config["decay_rate"],
            warmup_lr_init=optimizer_config["warmup_lr"],
            warmup_t=scheduler_config["warmup_epochs"],
            t_in_epochs=True,
        )
    elif scheduler_config["name"] == "multistep":
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            milestones=scheduler_config["multisteps"],
            gamma=scheduler_config["gamma"],
            warmup_lr_init=optimizer_config["warmup_lr"],
            warmup_t=scheduler_config["warmup_epochs"],
            t_in_epochs=True,
        )
    else:
        raise Exception(f"Unknown LR scheduler: {classifier_config['lr_scheduler']}")

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        t_initial: int,
        lr_min_rate: float,
        warmup_t=0,
        warmup_lr_init=0.0,
        t_in_epochs=True,
        noise_range_t=None,
        noise_pct=0.67,
        noise_std=1.0,
        noise_seed=42,
        initialize=True,
    ) -> None:
        super().__init__(
            optimizer,
            param_group_field="lr",
            noise_range_t=noise_range_t,
            noise_pct=noise_pct,
            noise_std=noise_std,
            noise_seed=noise_seed,
            initialize=initialize,
        )

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


class MultiStepLRScheduler(Scheduler):
    def __init__(
        self, optimizer: torch.optim.Optimizer, milestones, gamma=0.1, warmup_t=0, warmup_lr_init=0, t_in_epochs=True
    ) -> None:
        super().__init__(optimizer, param_group_field="lr")

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

        assert self.warmup_t <= min(self.milestones)

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            lrs = [v * (self.gamma ** bisect.bisect_right(self.milestones, t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None
