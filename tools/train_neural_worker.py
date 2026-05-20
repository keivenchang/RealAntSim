#!/usr/bin/env python3
"""Train a tiny worker-brain MLP on GPU.

By default this uses a synthetic teacher for smoke tests. For real training,
first run the Rust exporter and pass `--teacher-csv`; that trains by behavior
cloning on observations produced by the validated Classic sim brain.
Pickup/drop and pheromone deposit guards stay outside the net.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


OBS_DIM = 40
TURN_SCALE = 0.7
SENSOR_ANGLE = 0.6


class WorkerTurnNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
            nn.Linear(96, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs)).squeeze(-1) * TURN_SCALE


def unit_random(batch: int, device: torch.device) -> torch.Tensor:
    angle = torch.rand(batch, device=device) * 6.283185307179586
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)


def make_batch(batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    obs = torch.zeros(batch, OBS_DIM, device=device)
    carrying = (torch.rand(batch, device=device) < 0.45).float()
    wall = (torch.rand(batch, device=device) < 0.10).float()
    at_nest = (torch.rand(batch, device=device) < 0.03).float()
    pickup_dist = torch.rand(batch, device=device)
    sensors = torch.rand(batch, 9, device=device) * 8.0
    here = torch.rand(batch, 3, device=device) * 4.0
    food_grad = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    smell_grad = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    repel_grad = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    map_food_plan = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    map_home_plan = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    map_return = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    map_avoid = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    rough_home = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    path_home = unit_random(batch, device) * torch.rand(batch, 1, device=device)
    map_cell = torch.rand(batch, 3, device=device)
    has_walls = (torch.rand(batch, device=device) < 0.45).float()

    obs[:, 0] = carrying
    obs[:, 1] = wall
    obs[:, 2] = 1.0
    obs[:, 3] = 0.0
    obs[:, 4] = at_nest
    obs[:, 5] = pickup_dist
    obs[:, 6:15] = sensors / 8.0
    obs[:, 15:18] = here / 4.0
    obs[:, 18:20] = food_grad
    obs[:, 20:22] = smell_grad
    obs[:, 22:24] = repel_grad
    obs[:, 24:26] = map_food_plan
    obs[:, 26:28] = map_home_plan
    obs[:, 28:30] = map_return
    obs[:, 30:32] = map_avoid
    obs[:, 32:34] = rough_home
    obs[:, 34:36] = path_home
    obs[:, 36:39] = map_cell
    obs[:, 39] = has_walls

    left = torch.tensor(
        [torch.cos(torch.tensor(-SENSOR_ANGLE)), torch.sin(torch.tensor(-SENSOR_ANGLE))],
        device=device,
    )
    center = torch.tensor([1.0, 0.0], device=device)
    right = torch.tensor(
        [torch.cos(torch.tensor(SENSOR_ANGLE)), torch.sin(torch.tensor(SENSOR_ANGLE))],
        device=device,
    )
    dirs = torch.stack((left, center, right), dim=0)

    food_scores = torch.clamp(sensors[:, 0:3] - sensors[:, 6:9] * 2.0, min=0.0)
    home_scores = torch.clamp(sensors[:, 3:6] - sensors[:, 6:9] * 2.0, min=0.0)
    food_sensor = food_scores @ dirs
    home_sensor = home_scores @ dirs
    food_signal = food_scores.max(dim=1).values
    home_signal = home_scores.max(dim=1).values

    momentum = torch.tensor([1.0, 0.0], device=device).expand(batch, 2)
    left_wall = torch.tensor([0.0, -1.0], device=device)
    right_wall = torch.tensor([0.0, 1.0], device=device)
    wall_side = torch.where(
        (torch.arange(batch, device=device) % 2).unsqueeze(1) == 0,
        left_wall,
        right_wall,
    )
    wall_bias = wall.unsqueeze(1) * wall_side * 3.0

    outbound = (
        momentum * 0.65
        + food_sensor * 2.4
        + food_grad * 1.1
        + smell_grad * 2.2
        + map_food_plan * 2.4
        - repel_grad * 1.6
        + map_avoid * 0.15
        + wall_bias
    )
    carrier_follow = (
        home_sensor * 2.8
        + momentum * 0.65
        + map_home_plan * 1.4
        + map_return * 1.25
        + rough_home
        - repel_grad * 1.6
        + map_avoid * 0.15
        + wall_bias
    )
    carrier_search = (
        momentum * 1.4
        + map_home_plan * 1.4
        + map_return * 1.25
        + rough_home
        - smell_grad * (pickup_dist.unsqueeze(1) > 0.55).float() * 2.2
        - repel_grad * 1.6
        + map_avoid * 0.15
        + wall_bias
    )
    carrier = torch.where((home_signal >= 0.35).unsqueeze(1), carrier_follow, carrier_search)
    desired = torch.where(carrying.unsqueeze(1) > 0.5, carrier, outbound)
    target = torch.atan2(desired[:, 1], desired[:, 0]).clamp(-TURN_SCALE, TURN_SCALE)
    return obs, target


def load_teacher_csv(
    path: Path, device: torch.device, eval_fraction: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != OBS_DIM + 1:
        raise ValueError(f"expected {OBS_DIM + 1} CSV columns, got {data.shape[1]}")
    rng = np.random.default_rng(0)
    rng.shuffle(data)
    split = max(1, min(data.shape[0] - 1, int(data.shape[0] * (1.0 - eval_fraction))))
    train = data[:split]
    eval_ = data[split:]
    train_obs = torch.from_numpy(train[:, :OBS_DIM]).to(device)
    train_target = torch.from_numpy(train[:, OBS_DIM]).to(device)
    eval_obs = torch.from_numpy(eval_[:, :OBS_DIM]).to(device)
    eval_target = torch.from_numpy(eval_[:, OBS_DIM]).to(device)
    return train_obs, train_target, eval_obs, eval_target


def teacher_batch(
    obs: torch.Tensor, target: torch.Tensor, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.randint(0, obs.shape[0], (batch,), device=obs.device)
    return obs[idx], target[idx]


@torch.no_grad()
def mse(model: nn.Module, obs: torch.Tensor, target: torch.Tensor, batch: int) -> float:
    if obs.shape[0] == 0:
        return float("nan")
    take = min(batch, obs.shape[0])
    batch_obs, batch_target = teacher_batch(obs, target, take)
    pred = model(batch_obs)
    return float(torch.mean((pred - batch_target) ** 2).detach().cpu())


def weights_to_json(model: WorkerTurnNet) -> dict[str, object]:
    state = model.state_dict()
    return {
        "obs_dim": OBS_DIM,
        "turn_scale": TURN_SCALE,
        "layers": [
            {
                "weight": state["net.0.weight"].detach().cpu().tolist(),
                "bias": state["net.0.bias"].detach().cpu().tolist(),
                "activation": "tanh",
            },
            {
                "weight": state["net.2.weight"].detach().cpu().tolist(),
                "bias": state["net.2.bias"].detach().cpu().tolist(),
                "activation": "tanh",
            },
            {
                "weight": state["net.4.weight"].detach().cpu().tolist(),
                "bias": state["net.4.bias"].detach().cpu().tolist(),
                "activation": "tanh_scaled",
            },
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4_000)
    parser.add_argument("--batch", type=int, default=65_536)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--eval-fraction", type=float, default=0.10)
    parser.add_argument("--teacher-csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("run/neural_worker_weights.json"))
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = WorkerTurnNet().to(device)
    train_model: nn.Module = model
    if torch.cuda.device_count() >= 2:
        train_model = nn.DataParallel(model, device_ids=[0, 1])

    train_obs: torch.Tensor | None = None
    train_target: torch.Tensor | None = None
    eval_obs: torch.Tensor | None = None
    eval_target: torch.Tensor | None = None
    if args.teacher_csv is not None:
        train_obs, train_target, eval_obs, eval_target = load_teacher_csv(
            args.teacher_csv, device, args.eval_fraction
        )
        print(
            f"teacher_rows train={train_obs.shape[0]} eval={eval_obs.shape[0]} "
            f"path={args.teacher_csv}",
            flush=True,
        )

    opt = torch.optim.AdamW(train_model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss = torch.tensor(float("nan"), device=device)
    for step in range(1, args.steps + 1):
        if train_obs is not None and train_target is not None:
            obs, target = teacher_batch(train_obs, train_target, args.batch)
        else:
            obs, target = make_batch(args.batch, device)
        pred = train_model(obs)
        loss = torch.mean((pred - target) ** 2)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % 250 == 0:
            eval_loss = (
                mse(train_model, eval_obs, eval_target, args.batch)
                if eval_obs is not None and eval_target is not None
                else float("nan")
            )
            print(
                f"step={step} loss={loss.item():.6f} "
                f"eval_loss={eval_loss:.6f} device={device} "
                f"gpus={torch.cuda.device_count()}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = weights_to_json(model)
    payload["steps"] = args.steps
    payload["batch"] = args.batch
    payload["teacher_csv"] = str(args.teacher_csv) if args.teacher_csv is not None else None
    payload["final_loss"] = float(loss.detach().cpu())
    args.out.write_text(json.dumps(payload))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
