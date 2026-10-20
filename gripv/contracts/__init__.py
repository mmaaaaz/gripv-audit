"""Category contract registry. Add a module here when a category is ready."""
from __future__ import annotations

from .balance_scale_weighing import BalanceScale
from .conveyor_belt_sort import ConveyorBelt
from .robot_long_jump import RobotLongJump
from .bouncing_ball_restitution import BouncingBall
from .glasses_fill import GlassesFill
from .gear_train import GearTrain
from .temporal_flash_counting import TemporalFlash
from .inertia_race import InertiaRace
from .blinking_networks import BlinkingNetworks

CONTRACTS = {
    "glasses_fill": GlassesFill,
    "balance_scale_weighing": BalanceScale,
    "conveyor_belt_sort": ConveyorBelt,
    "robot_long_jump": RobotLongJump,
    "bouncing_ball_restitution": BouncingBall,
    "gear_train": GearTrain,
    "temporal_flash_counting": TemporalFlash,
    "inertia_race": InertiaRace,
    "blinking_networks": BlinkingNetworks,
}


def get(category: str):
    if category not in CONTRACTS:
        raise SystemExit(
            f"No contract for {category!r}. Ready: {', '.join(sorted(CONTRACTS))}"
        )
    return CONTRACTS[category]
