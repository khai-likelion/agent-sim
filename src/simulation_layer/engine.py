"""
Simulation engine for the Mangwon-dong commercial district.
Orchestrates the time loop and agent Perceive -> Decide -> React cycle.

Supports two modes:
1. StreetNetwork mode (primary): Agents move along actual street edges
2. Legacy H3 mode: Agents teleport to random locations in H3 grid
"""

import random
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List, Optional, Union

import pandas as pd

from config import get_settings
from src.data_layer.spatial_index import Environment
from src.data_layer.street_network import StreetNetwork, AgentLocation
from src.data_layer.population_stats import PopulationStatistics
from src.simulation_layer.persona.agent_persona import AgentPersona
from src.simulation_layer.persona.cognitive_modules.decide import DecideModule
from src.simulation_layer.persona.cognitive_modules.chained_decide import ChainedDecideModule
from src.simulation_layer.models import BusinessReport, SimulationEvent


class SimulationEngine:
    """
    Main simulation engine.
    Runs multi-day simulation with time-slot granularity.
    """

    WEEKDAY_KOR = ["월", "화", "수", "목", "금", "토", "일"]

    # 4-turn system: morning, lunch, dinner, night
    TIME_SLOTS_4TURN = {
        "아침": (6, 10),
        "점심": (11, 14),
        "저녁": (17, 21),
        "야간": (21, 24),
    }

    # Legacy 6-slot system
    TIME_SLOTS = {
        "00-06": (0, 6),
        "06-11": (6, 11),
        "11-14": (11, 14),
        "14-17": (14, 17),
        "17-21": (17, 21),
        "21-24": (21, 24),
    }

    def __init__(
        self,
        environment: Union[Environment, StreetNetwork],
        stats: PopulationStatistics,
        agents: List[AgentPersona],
        decide_module: Optional[DecideModule] = None,
    ):
        self.stats = stats
        self.agents = agents
        self.decide_module = decide_module or DecideModule()
        self.events: List[SimulationEvent] = []

        # Determine mode based on environment type
        if isinstance(environment, StreetNetwork):
            self.street_network = environment
            self.environment = None
            self.use_street_network = True
        else:
            self.street_network = None
            self.environment = environment
            self.use_street_network = False

    def _get_time_slot(self, hour: int, use_4turn: bool = False) -> str:
        """Get time slot name for given hour."""
        if use_4turn:
            for slot_name, (start, end) in self.TIME_SLOTS_4TURN.items():
                if start <= hour < end:
                    return slot_name
            # Map off-hours to closest slot
            if hour < 6:
                return "야간"
            elif hour < 11:
                return "아침"
            elif hour < 17:
                return "점심"
            else:
                return "저녁"
        else:
            for slot_name, (start, end) in self.TIME_SLOTS.items():
                if start <= hour < end:
                    return slot_name
            return "00-06"

    def _should_agent_be_active(self, weekday: str, time_slot: str) -> bool:
        weekday_weight = self.stats.get_weekday_weight(weekday)
        time_weight = self.stats.get_time_weight(time_slot)
        combined_weight = (weekday_weight + time_weight) / 2
        activity_prob = combined_weight * 7
        return random.random() < activity_prob

    def _get_random_start_location(self) -> tuple:
        """Get random start location (legacy H3 mode)."""
        settings = get_settings()
        lat = random.uniform(settings.area.lat_min, settings.area.lat_max)
        lng = random.uniform(settings.area.lng_min, settings.area.lng_max)
        return lat, lng

    def _initialize_agent_locations(self) -> None:
        """Initialize all agents with street network locations."""
        if not self.use_street_network:
            return

        settings = get_settings()
        for agent in self.agents:
            # Random start within area bounds
            lat = random.uniform(settings.area.lat_min, settings.area.lat_max)
            lng = random.uniform(settings.area.lng_min, settings.area.lng_max)
            agent.location = self.street_network.initialize_agent_location(lat, lng)

    def _move_agent_on_network(self, agent: AgentPersona, steps: int) -> AgentLocation:
        """Move agent along street network for given number of steps."""
        if agent.location is None:
            settings = get_settings()
            lat = random.uniform(settings.area.lat_min, settings.area.lat_max)
            lng = random.uniform(settings.area.lng_min, settings.area.lng_max)
            agent.location = self.street_network.initialize_agent_location(lat, lng)

        settings = get_settings()
        total_distance = steps * settings.simulation.agent_speed_mps * settings.simulation.movement_dt
        agent.location = self.street_network.move_agent(agent.location, total_distance)
        return agent.location

    def simulate_timestep(
        self,
        current_time: datetime,
        reports: Optional[List[BusinessReport]] = None,
    ) -> List[SimulationEvent]:
        """Run simulation for a single time step across all agents."""
        settings = get_settings()
        weekday = self.WEEKDAY_KOR[current_time.weekday()]

        # Use 4-turn system for ChainedDecideModule
        use_4turn = isinstance(self.decide_module, ChainedDecideModule)
        time_slot = self._get_time_slot(current_time.hour, use_4turn=use_4turn)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M")

        step_events = []

        for agent in self.agents:
            # 1. Activity check
            is_active = self._should_agent_be_active(weekday, time_slot)

            if not is_active:
                event = SimulationEvent(
                    timestamp=timestamp,
                    agent_id=agent.id,
                    agent_name=agent.name,
                    age_group=agent.age_group,
                    gender=agent.gender,
                    weekday=weekday,
                    time_slot=time_slot,
                    is_active=False,
                    current_lat=0,
                    current_lng=0,
                    visible_stores_count=0,
                    report_received=None,
                    decision="inactive",
                    decision_reason=f"{weekday} {time_slot} 시간대 비활동",
                    visited_store=None,
                    visited_category=None,
                )
                step_events.append(event)
                continue

            # 2. Perceive: location + visible stores
            if self.use_street_network:
                # Move agent along street network
                location = self._move_agent_on_network(
                    agent, settings.simulation.steps_per_timeslot
                )
                lat, lng = location.lat, location.lng
                visible_stores = self.street_network.get_nearby_stores(
                    location, k_ring=settings.simulation.k_ring
                )
            else:
                # Legacy H3 mode
                lat, lng = self._get_random_start_location()
                visible_stores = self.environment.get_visible_stores(lat, lng)

            # 3. Information: report reception (probabilistic)
            report = None
            report_received = None
            if reports and random.random() < settings.simulation.report_reception_probability:
                report = random.choice(reports)
                report_received = f"{report.store_name}: {report.description}"

            # 4. Decide: visit decision via cognitive module
            decision_result = self.decide_module.process(
                visible_stores, agent, report,
                time_slot=time_slot,
                weekday=weekday,
                memory_context="",  # TODO: integrate with EventMemory
            )

            # 5. Log event
            event = SimulationEvent(
                timestamp=timestamp,
                agent_id=agent.id,
                agent_name=agent.name,
                age_group=agent.age_group,
                gender=agent.gender,
                weekday=weekday,
                time_slot=time_slot,
                is_active=True,
                current_lat=lat,
                current_lng=lng,
                visible_stores_count=len(visible_stores),
                report_received=report_received,
                decision=decision_result["decision"],
                decision_reason=decision_result["decision_reason"],
                visited_store=decision_result["visited_store"],
                visited_category=decision_result["visited_category"],
            )
            step_events.append(event)

        return step_events

    def run_simulation(
        self,
        start_date: datetime,
        num_days: int | None = None,
        time_intervals_per_day: int | None = None,
        reports: Optional[List[BusinessReport]] = None,
    ) -> pd.DataFrame:
        """
        Run the full simulation.

        Returns:
            DataFrame of all simulation events.
        """
        settings = get_settings()
        num_days = num_days or settings.simulation.simulation_days
        time_intervals_per_day = (
            time_intervals_per_day or settings.simulation.time_slots_per_day
        )

        mode = "StreetNetwork" if self.use_street_network else "H3 Grid"
        print(f"Simulation start: {start_date.strftime('%Y-%m-%d')} ~ {num_days} days")
        print(f"  Mode: {mode}")
        print(f"  Agents: {len(self.agents)}")
        print(f"  Time intervals/day: {time_intervals_per_day}")
        if reports:
            print(f"  Business reports: {len(reports)}")
        print()

        # Initialize agent locations on street network
        if self.use_street_network:
            self._initialize_agent_locations()

        current_time = start_date
        all_events = []
        hour_interval = 24 // time_intervals_per_day

        for day in range(num_days):
            print(
                f"Day {day+1}/{num_days}: "
                f"{current_time.strftime('%Y-%m-%d')} "
                f"({self.WEEKDAY_KOR[current_time.weekday()]})"
            )

            # Reset daily agent states for ChainedDecideModule
            if isinstance(self.decide_module, ChainedDecideModule):
                self.decide_module.reset_daily_states()

            for interval in range(time_intervals_per_day):
                events = self.simulate_timestep(current_time, reports)
                all_events.extend(events)

                active_count = sum(1 for e in events if e.is_active)
                visit_count = sum(1 for e in events if e.decision == "visit")
                print(
                    f"  {current_time.strftime('%H:%M')} - "
                    f"active: {active_count}, visits: {visit_count}"
                )

                current_time += timedelta(hours=hour_interval)

        print(f"\nSimulation complete: {len(all_events)} total events")

        df = pd.DataFrame([asdict(event) for event in all_events])
        self.events = all_events
        return df
