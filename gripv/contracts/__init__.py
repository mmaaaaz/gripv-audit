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
from .chess_dance_trace import ChessDance
from .colored_route_puzzle import ColoredRoutePuzzle
from .moving_shadows import MovingShadows
from .buoyancy_tank import BuoyancyTank
from .multi_agent_chase import MultiAgentChase
from .rainbow_bagel import RainbowBagel
from .robot_archery import RobotArchery
from .sandpile_cascade import SandpileCascade
from .overlapping_shapes import OverlappingShapes
from .rolling_races import RollingRaces
from .shell_game_tracking import ShellGameTracking
from .swing_time import SwingTime
from .symmetry_pattern_video import SymmetryPattern
from .tangled_braid import TangledBraid
from .stacking_collapse import StackingCollapse
from .tile_alignment_shuffle import TileAlignmentShuffle
from .translucent_occlusion import TranslucentOcclusion
from .turntable_voxels import TurntableVoxels
from .vessels_and_clocks import VesselsAndClocks
from .well_flood_trace import WellFloodTrace

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
    "chess_dance_trace": ChessDance,
    "colored_route_puzzle": ColoredRoutePuzzle,
    "moving_shadows": MovingShadows,
    "buoyancy_tank": BuoyancyTank,
    "multi_agent_chase": MultiAgentChase,
    "rainbow_bagel": RainbowBagel,
    "robot_archery": RobotArchery,
    "sandpile_cascade": SandpileCascade,
    "overlapping_shapes": OverlappingShapes,
    "rolling_races": RollingRaces,
    "shell_game_tracking": ShellGameTracking,
    "swing_time": SwingTime,
    "symmetry_pattern_video": SymmetryPattern,
    "tangled_braid": TangledBraid,
    "stacking_collapse": StackingCollapse,
    "tile_alignment_shuffle": TileAlignmentShuffle,
    "translucent_occlusion": TranslucentOcclusion,
    "turntable_voxels": TurntableVoxels,
    "vessels_and_clocks": VesselsAndClocks,
    "well_flood_trace": WellFloodTrace,
}


def get(category: str):
    if category not in CONTRACTS:
        raise SystemExit(
            f"No contract for {category!r}. Ready: {', '.join(sorted(CONTRACTS))}"
        )
    return CONTRACTS[category]
