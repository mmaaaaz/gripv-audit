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
from .cellular_automata import CellularAutomata
from .object_tracking_mot import ObjectTracking

from .traffic_light_queue import TrafficLightQueue

from .mirror_beam_reflection import MirrorBeam

from .nested_shapes import NestedShapes

from .spinning_polyhedra import SpinningPolyhedra

from .shape_collision import ShapeCollision

from .simple_harmonic_motion import SimpleHarmonic
from .occluded_pattern_counting import OccludedPattern
from .hidden_mirrors import HiddenMirrors
from .spring_chain_waves import SpringChain

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
    "cellular_automata": CellularAutomata,
    "object_tracking_mot": ObjectTracking,
    "traffic_light_queue": TrafficLightQueue,
    "mirror_beam_reflection": MirrorBeam,
    "nested_shapes": NestedShapes,
    "spinning_polyhedra": SpinningPolyhedra,
    "shape_collision": ShapeCollision,
    "simple_harmonic_motion": SimpleHarmonic,
    "occluded_pattern_counting": OccludedPattern,
    "hidden_mirrors": HiddenMirrors,
    "spring_chain_waves": SpringChain,
}


def get(category: str):
    if category not in CONTRACTS:
        raise SystemExit(
            f"No contract for {category!r}. Ready: {', '.join(sorted(CONTRACTS))}"
        )
    return CONTRACTS[category]
