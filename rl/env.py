"""Gym-like environment wrapper for Tysiąc."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Set
import copy

from bots.base import BotStrategy
from bots.baseline_greedy import GreedyBot
from engine.cards import MARRIAGE_POINTS, Card, Rank, Suit
from engine.encode import encode_hand_binary, encode_trick, encode_seen_cards
from engine.game import GameSession, HandEngine, HandPhase
from engine.bidding import BASE_BID

from .action import ActionCatalog, ActionSpec, ActionType, BID_VALUES, CONTRACT_VALUES


@dataclass
class Observation:
    vector: List[float]
    legal_mask: List[int]
    info: Dict[str, object]


@dataclass
class RewardConfig:
    win: float = 1.0
    loss: float = -1.0
    draw: float = 0.0
    step: float = 0.0
    trick_win: float = 0.3
    trick_loss: float = -0.01
    meld_own: float = 0.2
    meld_opponent: float = -0.05
    contract_success: float = 1.2
    contract_fail: float = -0.3
    defend_contract_fail: float = 0.3
    defend_contract_success: float = -0.1
    proactive_bid: float = 0.1
    passive_bid_penalty: float = -0.05
    trump_lead_win: float = 0.05


class TysiacEnv:
    """Single-agent environment controlling player 0 against fixed opponents."""

    def __init__(
        self,
        *,
        opponent: Optional[BotStrategy] = None,
        opponent_factory: Optional[Callable[[], BotStrategy]] = None,
        agent_player: int = 0,
        starting_player: int = 0,
        seed: Optional[int] = None,
        reward_win: float = 1.0,
        reward_loss: float = -1.0,
        reward_step: float = 0.0,
        reward_config: Optional[RewardConfig] = None,
        include_seen_cards: bool = True,
    ) -> None:
        self.catalog = ActionCatalog()
        self._opponent_factory = opponent_factory
        self.opponent = opponent or (opponent_factory() if opponent_factory else GreedyBot())
        self._opponent_name = getattr(self.opponent, "name", type(self.opponent).__name__)
        if agent_player not in (0, 1):
            raise ValueError("agent_player must be 0 or 1.")
        self.agent_player = agent_player
        self.opponent_player = 1 - agent_player
        self.starting_player = starting_player
        self.seed = seed
        if reward_config is None:
            self.reward_config = RewardConfig(
                win=reward_win,
                loss=reward_loss,
                step=reward_step,
            )
        else:
            self.reward_config = reward_config

        self.session = GameSession(seed=seed)
        self.hand: Optional[HandEngine] = None
        self._obs_size: Optional[int] = None
        self._episode_stats: Dict[str, object] = {}
        self._last_trick_index: int = 0
        self._last_agent_meld_points: int = 0
        self._last_opponent_meld_points: int = 0
        self.include_seen_cards = include_seen_cards
        self._trump_bonus_awarded: Set[Suit] = set()

    def reset(self) -> Observation:
        if self._opponent_factory is not None:
            self.opponent = self._opponent_factory()
            self._opponent_name = getattr(self.opponent, "name", type(self.opponent).__name__)
        self.hand = self.session.start_hand(starting_player=self.starting_player)
        assert self.hand is not None
        if hasattr(self.opponent, "set_env_context"):
            self.opponent.set_env_context(
                catalog=self.catalog,
                session=self.session,
                agent_player=self.agent_player,
                opponent_player=self.opponent_player,
                include_seen_cards=self.include_seen_cards,
            )
        if hasattr(self.opponent, "on_hand_start"):
            self.opponent.on_hand_start(self.hand)
        self._reset_episode_stats()
        self._advance_others()
        obs = self._build_observation()
        self._obs_size = len(obs.vector)
        return obs

    def step(self, action_id: int) -> Tuple[Observation, float, bool, Dict[str, object]]:
        if self.hand is None:
            raise RuntimeError("Environment must be reset before stepping.")

        spec = self.catalog.spec(action_id)
        total_reward = 0.0

        if self.reward_config.step != 0.0:
            total_reward += self._apply_reward_components({"step": self.reward_config.step})

        self._apply_action(spec)
        components_after_action, meta_after_action = self._collect_shaped_rewards()
        total_reward += self._apply_reward_components(components_after_action)
        self._apply_meta_stats(meta_after_action)

        if self.hand.phase == HandPhase.COMPLETE:
            playing_player = self.hand.playing_player
            result = self.session.finish_hand()
            self.hand = None
            agent_score = result.new_scores[self.agent_player]
            opponent_score = result.new_scores[self.opponent_player]

            contract_components: Dict[str, float] = {}
            if playing_player is not None:
                if playing_player == self.agent_player:
                    self._episode_stats["contract_attempts"] += 1
                    if result.contract_success:
                        self._episode_stats["contract_success"] += 1
                        contract_components["contract"] = self.reward_config.contract_success
                    else:
                        contract_components["contract"] = self.reward_config.contract_fail
                else:
                    if result.contract_success:
                        contract_components["contract"] = self.reward_config.defend_contract_success
                    else:
                        contract_components["contract"] = self.reward_config.defend_contract_fail
            total_reward += self._apply_reward_components(contract_components)

            if agent_score > opponent_score:
                outcome_reward = self.reward_config.win
            elif agent_score < opponent_score:
                outcome_reward = self.reward_config.loss
            else:
                outcome_reward = self.reward_config.draw
            total_reward += self._apply_reward_components({"result": outcome_reward})

            done = True
            vector_length = self._obs_size or 0
            obs = Observation(
                vector=[0.0] * vector_length,
                legal_mask=[0] * self.catalog.size,
                info={"phase": "complete"},
            )
            episode_stats = copy.deepcopy(self._episode_stats)
            return obs, total_reward, done, {
                "scores": result.new_scores,
                "contract_success": result.contract_success,
                "defender_points_added": result.defender_points_added,
                "episode_stats": episode_stats,
            }

        self._advance_others()
        components_after_advance, meta_after = self._collect_shaped_rewards()
        total_reward += self._apply_reward_components(components_after_advance)
        self._apply_meta_stats(meta_after)

        obs = self._build_observation()
        done = False
        return obs, total_reward, done, {}

    # ------------------------------------------------------------------

    def _apply_action(self, spec: ActionSpec) -> None:
        assert self.hand is not None
        if spec.action_type == ActionType.NO_OP:
            return
        if self.hand.phase == HandPhase.AUCTION:
            self._apply_auction_action(spec)
        elif self.hand.phase == HandPhase.MUSIK:
            self._apply_musik_action(spec)
        elif self.hand.phase == HandPhase.PLAY:
            self._apply_play_action(spec)
        else:
            pass

    def _apply_auction_action(self, spec: ActionSpec) -> None:
        assert self.hand is not None
        if spec.action_type == ActionType.BID:
            bid_value = spec.payload[0]
            proof = self._proof_for_bid(bid_value)
            self.hand.bid(self.agent_player, bid_value, proof)
            if bid_value > (self.hand.auction.highest_bid or 0):
                if proof and self.reward_config.proactive_bid:
                    self._apply_reward_components({"bid": self.reward_config.proactive_bid})
        elif spec.action_type == ActionType.PASS:
            if self.reward_config.passive_bid_penalty and self._proof_for_bid(130, require_available=True):
                self._apply_reward_components({"bid": self.reward_config.passive_bid_penalty})
            self.hand.pass_bid(self.agent_player)
        else:
            raise ValueError(f"Invalid auction action {spec}")

    def _apply_musik_action(self, spec: ActionSpec) -> None:
        assert self.hand is not None
        if spec.action_type == ActionType.CHOOSE_MUSIK:
            self.hand.choose_musik(self.agent_player, index=spec.payload[0])
        elif spec.action_type == ActionType.RETURN_CARDS:
            cards = self._decode_card_pair(spec.payload)
            self.hand.return_to_unused(self.agent_player, cards=cards)
        elif spec.action_type == ActionType.SET_CONTRACT:
            self.hand.set_contract(spec.payload[0])
        else:
            raise ValueError(f"Invalid musik action {spec}")

    def _apply_play_action(self, spec: ActionSpec) -> None:
        assert self.hand is not None
        if spec.action_type != ActionType.PLAY_CARD:
            raise ValueError(f"Invalid play action {spec}")

        card = self._decode_card(spec.payload[0])
        partner = self._auto_meld_partner(card)
        declare = partner is not None
        self.hand.play_card(self.agent_player, card, declare_meld=declare, meld_partner=partner)

    # ------------------------------------------------------------------

    def _advance_others(self) -> None:
        """Let the opponent act until it's our turn or the hand completes."""
        if self.hand is None:
            return
        while True:
            if self.hand.phase == HandPhase.COMPLETE:
                break

            if self._current_player() == self.agent_player:
                break

            if self.hand.phase == HandPhase.AUCTION:
                decision = self.opponent.offer_bid(self.hand, self.opponent_player)
                if decision is None:
                    self.hand.pass_bid(self.opponent_player)
                else:
                    self.hand.bid(self.opponent_player, decision["bid"], decision.get("proof"))
            elif self.hand.phase == HandPhase.MUSIK:
                if self.hand.musik.chosen_index is None:
                    index = self.opponent.choose_musik(self.hand, self.opponent_player)
                    self.hand.choose_musik(self.opponent_player, index=index)
                elif len(self.hand.hands[self.opponent_player]) > 10:
                    cards = self.opponent.return_cards(self.hand, self.opponent_player)
                    self.hand.return_to_unused(self.opponent_player, cards=cards)
                elif self.hand.contract is None and self.hand.playing_player == self.opponent_player:
                    contract = self.opponent.choose_contract(self.hand, self.opponent_player)
                    self.hand.set_contract(contract)
            elif self.hand.phase == HandPhase.PLAY:
                card, declare, partner = self.opponent.play_card(self.hand, self.opponent_player)
                self.hand.play_card(self.opponent_player, card, declare_meld=declare, meld_partner=partner)
            else:
                break

    def _current_player(self) -> Optional[int]:
        if self.hand is None:
            return None
        if self.hand.phase == HandPhase.AUCTION:
            return self.hand.auction.current_player
        if self.hand.phase == HandPhase.MUSIK:
            return self.hand.playing_player
        if self.hand.phase == HandPhase.PLAY and self.hand.state is not None:
            return self.hand.state.current_player
        return None

    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        assert self.hand is not None
        base_vector = self._base_vector()
        legal_specs = self._legal_action_specs()
        mask = self.catalog.legal_mask(legal_specs)
        info = {
            "phase": self.hand.phase.name.lower(),
            "agent_player": self.agent_player,
            "current_player": self._current_player(),
        }
        return Observation(vector=base_vector, legal_mask=mask, info=info)

    def _base_vector(self) -> List[float]:
        assert self.hand is not None
        vector: List[float] = []

        if self.hand.state is not None:
            hand_cards = self.hand.state.hands[self.agent_player]
            vector.extend(encode_hand_binary(hand_cards))
            trick_cards = [play[1] for play in self.hand.state.current_trick.plays]
            vector.extend(encode_trick(trick_cards))
            trump_encoding = [0.0, 0.0, 0.0, 0.0, 0.0]
            if self.hand.state.trump is not None:
                trump_encoding[self.hand.state.trump.value] = 1.0
            else:
                trump_encoding[0] = 1.0
            vector.extend(trump_encoding)
            if self.include_seen_cards:
                seen_cards = set(hand_cards)
                for leader, lead_card, follower, follow_card, _winner in self.hand.state.trick_history:
                    seen_cards.add(lead_card)
                    seen_cards.add(follow_card)
                for _player, card in self.hand.state.current_trick.plays:
                    seen_cards.add(card)
                vector.extend(encode_seen_cards(seen_cards))
            vector.extend(
                [
                    float(self.hand.state.card_points[self.agent_player]),
                    float(self.hand.state.card_points[self.opponent_player]),
                ]
            )
            vector.extend(
                [
                    float(self.hand.state.meld_points[self.agent_player]),
                    float(self.hand.state.meld_points[self.opponent_player]),
                ]
            )
        else:
            hand_cards = self.hand.hands[self.agent_player]
            vector.extend(encode_hand_binary(hand_cards))
            vector.extend([-1.0, -1.0, -1.0, -1.0])
            vector.extend([1.0, 0.0, 0.0, 0.0, 0.0])
            vector.extend([0.0, 0.0])
            vector.extend([0.0, 0.0])
            if self.include_seen_cards:
                vector.extend(encode_seen_cards(hand_cards))

        # Suit distributions and meld potential flags
        declared_melds = set()
        if self.hand.state is not None:
            declared_melds = self.hand.state.melds_declared[self.agent_player]
        potential_meld_flags: List[float] = []
        suit_counts: List[float] = []
        hand_normalizer = 12.0
        for suit in Suit:
            count = sum(1 for card in hand_cards if card.suit is suit)
            suit_counts.append(float(count) / hand_normalizer)
            if suit in declared_melds:
                potential_meld_flags.append(0.0)
            else:
                has_king = any(card.suit is suit and card.rank is Rank.KING for card in hand_cards)
                has_queen = any(card.suit is suit and card.rank is Rank.QUEEN for card in hand_cards)
                potential_meld_flags.append(1.0 if has_king and has_queen else 0.0)

        current_trump_count = 0.0
        if self.hand.state is not None and self.hand.state.trump is not None:
            trump_suit = self.hand.state.trump
            current_trump_count = sum(1 for card in hand_cards if card.suit is trump_suit) / hand_normalizer
        vector.extend(potential_meld_flags)
        vector.extend(suit_counts)
        vector.append(current_trump_count)

        # Auction info
        highest_bid = self.hand.auction.highest_bid or 0
        vector.append(float(highest_bid))
        vector.append(float(self.hand.winning_bid or 0))
        vector.append(float(self.hand.contract or 0))
        if self.include_seen_cards:
            vector.append(float(self.hand.contract or 0) / 200.0)
        vector.append(float(len(self.hand.auction.history)))

        remaining_counts = self._remaining_cards()
        vector.extend([float(remaining_counts[0]), float(remaining_counts[1])])

        agent_score = float(self.session.scores[self.agent_player])
        opponent_score = float(self.session.scores[self.opponent_player])
        vector.extend([agent_score, opponent_score, agent_score - opponent_score])

        # Live point totals and contract context
        agent_live_points = agent_score
        opponent_live_points = opponent_score
        if self.hand.state is not None:
            agent_live_points += float(self.hand.state.card_points[self.agent_player] + self.hand.state.meld_points[self.agent_player])
            opponent_live_points += float(self.hand.state.card_points[self.opponent_player] + self.hand.state.meld_points[self.opponent_player])
        contract_value = float(self.hand.contract or 0)
        contract_needed = 0.0
        if contract_value > 0.0:
            contract_needed = max(contract_value - agent_live_points, 0.0) / 200.0
        score_margin = (agent_live_points - opponent_live_points) / 200.0
        vector.extend([
            agent_live_points / 1000.0,
            opponent_live_points / 1000.0,
            score_margin,
            contract_needed,
            1.0 if contract_value else 0.0,
        ])

        role_flags = [0.0, 0.0]
        if self.hand.playing_player is not None:
            if self.hand.playing_player == self.agent_player:
                role_flags = [1.0, 0.0]
            else:
                role_flags = [0.0, 1.0]
        vector.extend(role_flags)

        tricks_played = 0.0
        if self.hand.state is not None:
            tricks_played = float(len(self.hand.state.trick_history))
        vector.append(tricks_played)
        total_tricks = 10.0
        tricks_remaining = (total_tricks - tricks_played) / total_tricks
        cards_remaining_agent = float(len(hand_cards)) / 12.0
        vector.extend([tricks_remaining, cards_remaining_agent])

        vector.extend(self._encode_auction_history(limit=6))

        # Phase indicator
        phase_one_hot = [0.0, 0.0, 0.0, 0.0]
        phase_index = {
            HandPhase.AUCTION: 0,
            HandPhase.MUSIK: 1,
            HandPhase.PLAY: 2,
            HandPhase.COMPLETE: 3,
        }[self.hand.phase]
        phase_one_hot[phase_index] = 1.0
        vector.extend(phase_one_hot)

        return vector

    def _legal_action_specs(self) -> Iterable[ActionSpec]:
        assert self.hand is not None
        if self.hand.phase == HandPhase.AUCTION:
            actions = list(self._auction_actions())
        elif self.hand.phase == HandPhase.MUSIK:
            actions = list(self._musik_actions())
        elif self.hand.phase == HandPhase.PLAY:
            actions = list(self._play_actions())
        else:
            actions = []
        if not actions:
            actions = [ActionSpec(ActionType.NO_OP, ())]
        return actions

    @property
    def opponent_name(self) -> str:
        return self._opponent_name

    def _auction_actions(self) -> Iterable[ActionSpec]:
        assert self.hand is not None
        if self.hand.auction.current_player != self.agent_player:
            return []
        legal: List[ActionSpec] = []
        highest = self.hand.auction.highest_bid or 0
        for bid in BID_VALUES:
            if bid > highest and self._proof_for_bid(bid, require_available=True):
                legal.append(ActionSpec(ActionType.BID, (bid,)))
        if (
            self.hand.auction.highest_bid is not None
            and self.hand.auction.highest_bidder != self.agent_player
        ):
            legal.append(ActionSpec(ActionType.PASS, ()))
        return legal

    def _musik_actions(self) -> Iterable[ActionSpec]:
        assert self.hand is not None
        legal: List[ActionSpec] = []
        if self.hand.playing_player != self.agent_player:
            return legal
        if self.hand.musik.chosen_index is None:
            legal.extend(ActionSpec(ActionType.CHOOSE_MUSIK, (idx,)) for idx in (0, 1))
        elif len(self.hand.hands[self.agent_player]) > 10:
            combos = self._card_pair_specs(self.hand.hands[self.agent_player])
            legal.extend(combos)
        else:
            minimum = self.hand.winning_bid or 100
            default_contract = max(minimum, ((minimum + 9) // 10) * 10)
            legal.append(ActionSpec(ActionType.SET_CONTRACT, (default_contract,)))
        return legal

    def _play_actions(self) -> Iterable[ActionSpec]:
        assert self.hand is not None and self.hand.state is not None
        if self.hand.state.current_player != self.agent_player:
            return []
        legal_cards = self.hand.state.available_moves(self.agent_player)
        return [
            ActionSpec(ActionType.PLAY_CARD, (self.catalog.encode_card(card),))
            for card in legal_cards
        ]

    def _card_pair_specs(self, hand_cards: List[Card]) -> Iterable[ActionSpec]:
        encoded = sorted((self.catalog.encode_card(card), card) for card in hand_cards)
        specs = []
        for i in range(len(encoded)):
            for j in range(i + 1, len(encoded)):
                pair_ids = (encoded[i][0], encoded[j][0])
                specs.append(ActionSpec(ActionType.RETURN_CARDS, pair_ids))
        return specs

    def _decode_card(self, card_id: int) -> Card:
        # Reverse mapping using HandService view
        assert self.hand is not None
        # Search in hand for matching encoded id
        card_sources: Sequence[Card]
        if self.hand.state:
            card_sources = self.hand.state.hands[self.agent_player]
        else:
            card_sources = self.hand.hands[self.agent_player]
        for card in card_sources:
            if self.catalog.encode_card(card) == card_id:
                return card
        raise ValueError(f"Card id {card_id} not found in hand.")

    def _auto_meld_partner(self, card: Card) -> Optional[Card]:
        if self.hand is None or self.hand.state is None:
            return None
        state = self.hand.state
        if not state.current_trick.is_empty():
            return None
        if card.rank not in (Rank.KING, Rank.QUEEN):
            return None
        if card.suit in state.melds_declared[self.agent_player]:
            return None
        partner_rank = Rank.QUEEN if card.rank is Rank.KING else Rank.KING
        for candidate in state.hands[self.agent_player]:
            if candidate is card:
                continue
            if candidate.suit is card.suit and candidate.rank is partner_rank:
                return candidate
        return None

    def _decode_card_pair(self, pair: Tuple[int, int]) -> List[Card]:
        return [self._decode_card(idx) for idx in pair]

    def _remaining_cards(self) -> Tuple[int, int]:
        if self.hand is None:
            return (0, 0)
        if self.hand.state:
            return (
                len(self.hand.state.hands[self.agent_player]),
                len(self.hand.state.hands[self.opponent_player]),
            )
        return (
            len(self.hand.hands[self.agent_player]),
            len(self.hand.hands[self.opponent_player]),
        )

    def _encode_auction_history(self, limit: int = 6) -> List[float]:
        assert self.hand is not None
        history = self.hand.auction.history[-limit:]
        encoded: List[float] = []
        for player, action, amount in history:
            encoded.extend(
                [
                    1.0 if player == self.agent_player else 0.0,
                    1.0 if player == self.opponent_player else 0.0,
                ]
            )
            action_one_hot = [0.0, 0.0, 0.0]
            if action == "bid":
                action_one_hot[0] = 1.0
            elif action == "pass":
                action_one_hot[1] = 1.0
            else:
                action_one_hot[2] = 1.0
            encoded.extend(action_one_hot)
            encoded.append(float(amount or 0))

        expected_len = limit * (2 + 3 + 1)
        if len(encoded) < expected_len:
            encoded.extend([0.0] * (expected_len - len(encoded)))
        return encoded

    def _agent_hand_cards(self) -> Sequence[Card]:
        assert self.hand is not None
        if self.hand.state:
            return self.hand.state.hands[self.agent_player]
        return self.hand.hands[self.agent_player]

    def _proof_for_bid(self, bid: int, require_available: bool = False):
        required = bid - BASE_BID
        if required <= 0:
            return True if require_available else None
        cards = list(self._agent_hand_cards())
        for suit, points in MARRIAGE_POINTS.items():
            if points != required:
                continue
            king = next((card for card in cards if card.suit is suit and card.rank is Rank.KING), None)
            queen = next((card for card in cards if card.suit is suit and card.rank is Rank.QUEEN), None)
            if king and queen:
                proof = {"suit": suit, "cards": {king, queen}}
                return proof if not require_available else True
        return False if require_available else None

    def _reset_episode_stats(self) -> None:
        self._episode_stats = {
            "tricks_won": 0,
            "tricks_lost": 0,
            "agent_melds": 0,
            "opponent_melds": 0,
            "contract_attempts": 0,
            "contract_success": 0,
            "reward_total": 0.0,
            "reward_breakdown": {
                "step": 0.0,
                "trick": 0.0,
                "meld": 0.0,
                "contract": 0.0,
                "result": 0.0,
            },
        }
        self._last_trick_index = 0
        self._last_agent_meld_points = 0
        self._last_opponent_meld_points = 0
        self._trump_bonus_awarded = set()

    def _apply_reward_components(self, components: Dict[str, float]) -> float:
        if not components:
            return 0.0
        total = sum(components.values())
        if total == 0.0:
            return 0.0
        self._episode_stats["reward_total"] += total
        breakdown: Dict[str, float] = self._episode_stats["reward_breakdown"]  # type: ignore[assignment]
        for key, value in components.items():
            breakdown[key] = breakdown.get(key, 0.0) + value
        return total

    def _collect_shaped_rewards(self) -> Tuple[Dict[str, float], Dict[str, int]]:
        reward_components: Dict[str, float] = {}
        meta: Dict[str, int] = {}
        if self.hand is None or self.hand.state is None:
            return reward_components, meta
        reward_components, meta = self._process_tricks(reward_components, meta)
        reward_components, meta = self._process_melds(reward_components, meta)
        return reward_components, meta

    def _process_tricks(
        self,
        reward_components: Dict[str, float],
        meta: Dict[str, int],
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        assert self.hand is not None and self.hand.state is not None
        new_tricks = self.hand.state.trick_history[self._last_trick_index :]
        self._last_trick_index = len(self.hand.state.trick_history)
        for trick in new_tricks:
            winner = trick[4]
            leader = trick[0]
            lead_card = trick[1]
            if winner == self.agent_player:
                reward_components["trick"] = reward_components.get("trick", 0.0) + self.reward_config.trick_win
                meta["tricks_won"] = meta.get("tricks_won", 0) + 1
                if (
                    self.reward_config.trump_lead_win
                    and leader == self.agent_player
                    and lead_card.rank in (Rank.KING, Rank.QUEEN)
                    and self.hand is not None
                    and self.hand.state is not None
                    and self.hand.state.trump is not None
                    and self.hand.state.trump == lead_card.suit
                    and lead_card.suit not in self._trump_bonus_awarded
                ):
                    reward_components["trump_lead"] = reward_components.get("trump_lead", 0.0) + self.reward_config.trump_lead_win
                    self._trump_bonus_awarded.add(lead_card.suit)
            else:
                reward_components["trick"] = reward_components.get("trick", 0.0) + self.reward_config.trick_loss
                meta["tricks_lost"] = meta.get("tricks_lost", 0) + 1
        return reward_components, meta

    def _process_melds(
        self,
        reward_components: Dict[str, float],
        meta: Dict[str, int],
    ) -> Tuple[Dict[str, float], Dict[str, int]]:
        assert self.hand is not None and self.hand.state is not None
        agent_meld = self.hand.state.meld_points[self.agent_player]
        opponent_meld = self.hand.state.meld_points[self.opponent_player]

        if agent_meld > self._last_agent_meld_points:
            reward_components["meld"] = reward_components.get("meld", 0.0) + self.reward_config.meld_own
            meta["agent_melds"] = meta.get("agent_melds", 0) + 1
            self._last_agent_meld_points = agent_meld
        if opponent_meld > self._last_opponent_meld_points:
            reward_components["meld"] = reward_components.get("meld", 0.0) + self.reward_config.meld_opponent
            meta["opponent_melds"] = meta.get("opponent_melds", 0) + 1
            self._last_opponent_meld_points = opponent_meld
        return reward_components, meta

    def _apply_meta_stats(self, meta: Dict[str, int]) -> None:
        if not meta:
            return
        for key, value in meta.items():
            if key in self._episode_stats:
                self._episode_stats[key] += value
