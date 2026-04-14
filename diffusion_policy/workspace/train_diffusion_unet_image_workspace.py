if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import sys
import json
import subprocess
import tempfile
import hydra
from typing import Optional
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _should_run_rollout(epoch_0based: int, rollout_every: int) -> bool:
    """
    Rollout on 1-based epochs: 1, rollout_every, 2*rollout_every, ... (e.g. 1,30,60,...,150).
    With num_epochs=150 and rollout_every=30 this includes the final epoch (150), not only 0,30,...,120.
    """
    if rollout_every <= 0:
        return False
    e1 = epoch_0based + 1
    return (e1 == 1) or (e1 % rollout_every == 0)


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def _run_rollout_subprocess(self, cfg, policy) -> dict:
        """
        Run env_runner in a fresh subprocess to avoid EGL/CUDA driver conflicts.
        The child process has no inherited CUDA context, so MuJoCo EGL can
        initialise cleanly before PyTorch touches the GPU.
        """
        _EVAL_SCRIPT = (
            pathlib.Path(__file__).resolve().parents[2] / "scripts" / "rollout_eval.py"
        )
        out = pathlib.Path(self.output_dir).resolve()
        tmp = out / ".rollout_tmp"
        tmp.mkdir(parents=True, exist_ok=True)

        # Serialise runner config and policy config
        runner_cfg_json = tmp / "runner_cfg.json"
        policy_cfg_json = tmp / "policy_cfg.json"
        policy_state_pt = tmp / "policy_state.pt"
        result_json = (tmp / "result.json").resolve()

        runner_cfg_json.write_text(
            json.dumps(OmegaConf.to_container(cfg.task.env_runner, resolve=True)))
        policy_cfg_json.write_text(
            json.dumps(OmegaConf.to_container(cfg.policy, resolve=True)))

        # Rollout-phase obs occlusion config (training data is always clean)
        rollout_occ_cfg = OmegaConf.to_container(
            cfg.training.get("rollout_obs_occlusion", {}), resolve=True) or {}
        occ_cfg_json = json.dumps(rollout_occ_cfg)

        # Save policy state dict (includes normalizer registered as sub-module)
        torch.save(policy.state_dict(), str(policy_state_pt))

        env = {**os.environ,
               "MUJOCO_GL": "osmesa",            # software renderer; avoids EGL segfault on HPC
               "MUJOCO_PY_FORCE_CPU": "1",       # force CPU (osmesa) extension on GPU nodes
               "MESA_GL_VERSION_OVERRIDE": "3.3", # MuJoCo needs OpenGL 3.3+
               "MESA_GLSL_VERSION_OVERRIDE": "330",
               "DP_ROBOMIMIC_NO_EGL_PROBE": "1",
               "DP_ROBOSUITE_RENDER_GPU_DEVICE_ID": "0",
               "PYTHONUNBUFFERED": "1"}
        if not env.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = "0"

        cmd = [
            sys.executable, str(_EVAL_SCRIPT),
            "--runner-cfg",        str(runner_cfg_json.resolve()),
            "--policy-cfg",        str(policy_cfg_json.resolve()),
            "--policy-state",      str(policy_state_pt.resolve()),
            "--output-dir",        str(out),
            "--result-json",       str(result_json),
            "--device",            str(cfg.training.device),
            "--obs-occlusion-cfg", occ_cfg_json,
        ]
        # Slurm에서 자식 stdout/stderr가 TTY가 아니면 버퍼링될 수 있음 → PYTHONUNBUFFERED + flush
        print(f"[subprocess rollout] cwd={os.getcwd()}", flush=True)
        print(f"[subprocess rollout] result_json={result_json}", flush=True)
        print(f"[subprocess rollout] cmd={' '.join(cmd)}", flush=True)
        try:
            probe = tmp / ".quota_probe"
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            raise RuntimeError(
                f"Cannot write under rollout tmp dir (disk full / quota / permission?): {tmp} ({e})"
            ) from e

        print(f"[subprocess rollout] launching {_EVAL_SCRIPT.name} …", flush=True)
        ret = subprocess.run(cmd, env=env, stdout=sys.stdout, stderr=sys.stderr)
        print(f"[subprocess rollout] child returncode={ret.returncode}", flush=True)

        if result_json.is_file():
            raw_text = result_json.read_text()
            try:
                runner_log = json.loads(raw_text)
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Invalid JSON in {result_json}: {e}\n---\n{raw_text[:4000]}"
                ) from e
            if isinstance(runner_log, dict) and "rollout_fatal" in runner_log:
                raise RuntimeError(
                    f"rollout_eval.py crashed:\n{runner_log.get('rollout_fatal', '')}"
                )
            if ret.returncode != 0:
                raise RuntimeError(
                    f"rollout subprocess exit {ret.returncode} (unexpected with metrics file); "
                    f"body={raw_text[:4000]}"
                )
            print(f"[subprocess rollout] results: {runner_log}", flush=True)
            return runner_log

        raise FileNotFoundError(
            f"Rollout subprocess rc={ret.returncode} but {result_json} missing. "
            f"Look in .err for 'rollout_eval:' (child start) and 'subprocess rollout cmd='. "
            f"Typical: quota/full disk, rm -rf on this output dir during rollout, or wrong PYTHON/script path."
        )

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Rollout strategy:
        #   use_subprocess_rollout=True  → spawn fresh child process per rollout (no EGL/CUDA conflict)
        #   use_subprocess_rollout=False → direct env_runner in this process (may segfault on some nodes)
        #   skip_env_runner=True         → no rollout at all
        use_subprocess_rollout = cfg.training.get("use_subprocess_rollout", False)
        env_runner: Optional[BaseImageRunner] = None
        if not cfg.training.get("skip_env_runner", False) and not use_subprocess_rollout:
            env_runner = hydra.utils.instantiate(
                cfg.task.env_runner,
                output_dir=self.output_dir)
            assert isinstance(env_runner, BaseImageRunner)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        # use val_loss when: no rollout at all, OR subprocess rollout (may fail intermittently)
        # use test_mean_score only when env_runner runs directly in this process (reliable)
        topk_cfg = dict(
            OmegaConf.to_container(cfg.checkpoint.topk, resolve=True))
        no_reliable_score = (
            cfg.training.get("skip_env_runner", False) or use_subprocess_rollout
        )
        if no_reliable_score:
            topk_cfg["monitor_key"] = "val_loss"
            topk_cfg["mode"] = "min"
            topk_cfg["format_str"] = (
                "epoch={epoch:04d}-val_loss={val_loss:.4f}.ckpt")
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **topk_cfg
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            # Smoke / quick curve: was 2 epochs × 3 train & val steps; ~4× for denser logs.json / graphs
            cfg.training.num_epochs = 8
            cfg.training.max_train_steps = 12
            cfg.training.max_val_steps = 12
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                # 1-based epoch in logs/plots (epoch 1 .. num_epochs)
                epoch_1based = self.epoch + 1
                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    self.model.obs_encoder.eval()
                    self.model.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {epoch_1based}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': epoch_1based,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout (1-based schedule: 1, rollout_every, 2*rollout_every, ...)
                if _should_run_rollout(self.epoch, cfg.training.rollout_every):
                    if use_subprocess_rollout:
                        runner_log = self._run_rollout_subprocess(cfg, policy)
                        step_log.update(runner_log)
                    elif env_runner is not None:
                        runner_log = env_runner.run(policy)
                        step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {epoch_1based}", 
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = batch['obs']
                        gt_action = batch['action']
                        
                        result = policy.predict_action(obs_dict)
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
