"""Self-play bot that mirrors a saved PPO policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.distributions import Categorical

from bots.base import BotStrategy
from engine.cards import MARRIAGE_POINTS, Card, Rank, Suit
from engine.bidding import BASE_BID
from engine.encode import encode_hand_binary, encode_seen_cards, encode_trick
from engine.game import HandEngine, HandPhase
from rl.action import ActionCatalog, ActionSpec, ActionType, BID_VALUES
from rl.ppo_lstm.net import build_policy_value_net


@dataclass
class _ObservationView:
    vector: List[float]
    legal_mask: List[int]


def _adapt_state_dict(state_dict: Dict[str, torch.Tensor], model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Pad legacy checkpoints when observation or action dimensions grow."""
    adapted_state = dict(state_dict)
    model_state = model.state_dict()

    def _adapt_tensor(key: str, allow_expand_cols: bool = False) -> None:
        if key not in adapted_state or key not in model_state:
            return
        old_tensor = adapted_state[key]
        new_tensor = model_state[key]
        if old_tensor.shape == new_tensor.shape:
            return
        if old_tensor.ndim == 2 and new_tensor.ndim == 2:
            if old_tensor.shape[0] > new_tensor.shape[0]:
                raise RuntimeError(f"Cannot adapt tensor {key}: rows {old_tensor.shape[0]} > {new_tensor.shape[0]}.")
            if old_tensor.shape[1] > new_tensor.shape[1]:
                raise RuntimeError(f"Cannot adapt tensor {key}: cols {old_tensor.shape[1]} > {new_tensor.shape[1]}.")
            if not allow_expand_cols and old_tensor.shape[1] != new_tensor.shape[1]:
                raise RuntimeError(
                    f"Cannot adapt tensor {key}: column mismatch {old_tensor.shape[1]} vs {new_tensor.shape[1]}."
                )
            padded = new_tensor.clone()
            padded[: old_tensor.shape[0], : old_tensor.shape[1]] = old_tensor
            adapted_state[key] = padded
        elif old_tensor.ndim == 1 and new_tensor.ndim == 1:
            if old_tensor.shape[0] > new_tensor.shape[0]:
                raise RuntimeError(f"Cannot adapt tensor {key}: length {old_tensor.shape[0]} > {new_tensor.shape[0]}.")
            padded = new_tensor.clone()
            padded[: old_tensor.shape[0]] = old_tensor
            adapted_state[key] = padded
        else:
            raise RuntimeError(f"Incompatible tensor ranks for {key}: {old_tensor.ndim} vs {new_tensor.ndim}.")

    # Observation encoder (allow extra input features)
    _adapt_tensor("obs_encoder.0.weight", allow_expand_cols=True)
    _adapt_tensor("obs_encoder.0.bias")

    # Policy head variants (legacy and current architecture)
    _adapt_tensor("policy_head.weight")
    _adapt_tensor("policy_head.bias")
    _adapt_tensor("policy_head.0.weight")
    _adapt_tensor("policy_head.0.bias")
    _adapt_tensor("policy_head.2.weight")
    _adapt_tensor("policy_head.2.bias")

    return adapted_state


class SelfPlayBot(BotStrategy):
    """Bot that replays a frozen PPO policy for self-play opponents."""

    def __init__(self, checkpoint_path: Path | str, device: str = "cpu") -> None:
        from torch.serialization import add_safe_globals
        from pathlib import PosixPath

        add_safe_globals([PosixPath])

        self._checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.name = f"SelfPlay({self._checkpoint_path.stem})"

        self._policy_catalog = ActionCatalog()
        checkpoint = torch.load(self._checkpoint_path, map_location=self.device, weights_only=False)
        raw_config = checkpoint.get("config")
        if isinstance(raw_config, dict):
            policy_include_seen_cards = raw_config.get("include_seen_cards", True)
        else:
            policy_include_seen_cards = getattr(raw_config, "include_seen_cards", True)

        state_dict: Dict[str, torch.Tensor] = checkpoint["model_state"]
        obs_weight = state_dict["obs_encoder.0.weight"]
        lstm_weight = state_dict["lstm.weight_ih_l0"]
        obs_size = obs_weight.shape[1]
        hidden_size = obs_weight.shape[0]
        lstm_size = lstm_weight.shape[0] // 4
        action_size = self._policy_catalog.size

        legacy = checkpoint.get("use_legacy_model", False) or (hidden_size == 256 and lstm_size == 128)

        self.model = build_policy_value_net(
            obs_size=obs_size,
            action_size=action_size,
            hidden_size=hidden_size,
            lstm_size=lstm_size,
            legacy=legacy,
        ).to(self.device)
        adapted_state = _adapt_state_dict(state_dict, self.model)
        self.model.load_state_dict(adapted_state)
        self.model.eval()

        self.policy_include_seen_cards = policy_include_seen_cards
        self.hidden = self.model.initial_state(device=self.device)

        # Runtime context (populated via set_env_context)
        self.catalog: ActionCatalog = self._policy_catalog
        self.session = None
        self.agent_player: int = 0
        self.player: int = 1
        self.include_seen_cards: bool = self.policy_include_seen_cards

    # ------------------------------------------------------------------ context hooks

    def set_env_context(
        self,
        *,
        catalog: ActionCatalog,
        session,
        agent_player: int,
        opponent_player: int,
        include_seen_cards: bool,
    ) -> None:
        self.catalog = catalog
        self.session = session
        self.agent_player = agent_player
        self.player = opponent_player
        self.include_seen_cards = include_seen_cards and self.policy_include_seen_cards
        self.hidden = self.model.initial_state(device=self.device)

    def on_hand_start(self, hand: HandEngine) -> None:
        self.hidden = self.model.initial_state(device=self.device)

    # ------------------------------------------------------------------ BotStrategy impl

    def offer_bid(self, hand: HandEngine, player: int):
        spec = self._select_action(hand)
        if spec.action_type == ActionType.BID:
            bid_value = spec.payload[0]
            proof = self._proof_for_bid(hand, self.player, bid_value)
            return {"bid": bid_value, "proof": proof}
        if spec.action_type == ActionType.PASS:
            return None
        raise RuntimeError(f"SelfPlayBot received invalid auction action {spec}")

    def choose_musik(self, hand: HandEngine, player: int) -> int:
        spec = self._select_action(hand)
        if spec.action_type != ActionType.CHOOSE_MUSIK:
            raise RuntimeError(f"SelfPlayBot expected CHOOSE_MUSIK action, got {spec}")
        return spec.payload[0]

    def return_cards(self, hand: HandEngine, player: int) -> Sequence[Card]:
        spec = self._select_action(hand)
        if spec.action_type != ActionType.RETURN_CARDS:
            raise RuntimeError(f"SelfPlayBot expected RETURN_CARDS action, got {spec}")
        return self._decode_card_pair(hand, spec.payload)

    def choose_contract(self, hand: HandEngine, player: int) -> int:
        spec = self._select_action(hand)
        if spec.action_type != ActionType.SET_CONTRACT:
            raise RuntimeError(f"SelfPlayBot expected SET_CONTRACT action, got {spec}")
        return spec.payload[0]

    def play_card(
        self, hand: HandEngine, player: int
    ) -> Tuple[Card, bool, Optional[Card]]:
        spec = self._select_action(hand)
        if spec.action_type != ActionType.PLAY_CARD:
            raise RuntimeError(f"SelfPlayBot expected PLAY_CARD action, got {spec}")
        card = self._decode_card(hand, spec.payload[0])
        partner = self._auto_meld_partner(hand, card)
        declare = partner is not None
        return card, declare, partner

    # ------------------------------------------------------------------ action selection

    def _select_action(self, hand: HandEngine) -> ActionSpec:
        obs = self._build_observation(hand)
        expected = self.model.obs_encoder[0].in_features  # type: ignore[index]
        vector = obs.vector
        if len(vector) < expected:
            vector = vector + [0.0] * (expected - len(vector))
        elif len(vector) > expected:
            vector = vector[:expected]
        obs_tensor = torch.tensor(vector, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_tensor = torch.tensor(obs.legal_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(obs_tensor, self.hidden)
        self.hidden = outputs.hidden
        logits = torch.where(
            mask_tensor,
            outputs.logits,
            torch.tensor(-1e9, device=self.device),
        )
        dist = Categorical(logits=logits)
        action_id = torch.argmax(dist.probs, dim=-1).item()
        spec = self.catalog.spec(action_id)
        return spec

    # ------------------------------------------------------------------ observation helpers

    def _build_observation(self, hand: HandEngine) -> _ObservationView:
        vector: List[float] = []
        player = self.player
        opponent = 1 - player
        state = hand.state

        if state is not None:
            hand_cards = state.hands[player]
            vector.extend(encode_hand_binary(hand_cards))
            trick_cards = [play[1] for play in state.current_trick.plays]
            vector.extend(encode_trick(trick_cards))
            trump_encoding = [0.0, 0.0, 0.0, 0.0, 0.0]
            if state.trump is not None:
                trump_encoding[state.trump.value] = 1.0
            else:
                trump_encoding[0] = 1.0
            vector.extend(trump_encoding)
            if self.include_seen_cards:
                seen_cards = set(hand_cards)
                for _, lead_card, _, follow_card, _ in state.trick_history:
                    seen_cards.add(lead_card)
                    seen_cards.add(follow_card)
                for _, card in state.current_trick.plays:
                    seen_cards.add(card)
                vector.extend(encode_seen_cards(seen_cards))
            vector.extend(
                [
                    float(state.card_points[player]),
                    float(state.card_points[opponent]),
                ]
            )
            vector.extend(
                [
                    float(state.meld_points[player]),
                    float(state.meld_points[opponent]),
                ]
            )
        else:
            hand_cards = hand.hands[player]
            vector.extend(encode_hand_binary(hand_cards))
            vector.extend([-1.0, -1.0, -1.0, -1.0])
            vector.extend([1.0, 0.0, 0.0, 0.0, 0.0])
            vector.extend([0.0, 0.0])
            vector.extend([0.0, 0.0])
            if self.include_seen_cards:
                vector.extend(encode_seen_cards(hand_cards))

        highest_bid = hand.auction.highest_bid or 0
        vector.append(float(highest_bid))
        vector.append(float(hand.winning_bid or 0))
        vector.append(float(hand.contract or 0))
        if self.include_seen_cards:
            vector.append(float(hand.contract or 0) / 200.0)
        vector.append(float(len(hand.auction.history)))

        remaining_counts = self._remaining_cards(hand, player, opponent)
        vector.extend([float(remaining_counts[0]), float(remaining_counts[1])])

        if self.session is not None:
            agent_score = float(self.session.scores[player])
            opponent_score = float(self.session.scores[opponent])
        else:
            agent_score = 0.0
            opponent_score = 0.0
        vector.extend([agent_score, opponent_score, agent_score - opponent_score])

        role_flags = [0.0, 0.0]
        if hand.playing_player is not None:
            if hand.playing_player == player:
                role_flags = [1.0, 0.0]
            else:
                role_flags = [0.0, 1.0]
        vector.extend(role_flags)

        tricks_played = 0.0
        if state is not None:
            tricks_played = float(len(state.trick_history))
        vector.append(tricks_played)

        vector.extend(self._encode_auction_history(hand, player, opponent, limit=6))

        phase_one_hot = [0.0, 0.0, 0.0, 0.0]
        phase_index = {
            HandPhase.AUCTION: 0,
            HandPhase.MUSIK: 1,
            HandPhase.PLAY: 2,
            HandPhase.COMPLETE: 3,
        }[hand.phase]
        phase_one_hot[phase_index] = 1.0
        vector.extend(phase_one_hot)

        legal_specs = list(self._legal_action_specs(hand, player))
        if not legal_specs:
            legal_specs = [ActionSpec(ActionType.NO_OP, ())]
        legal_mask = self.catalog.legal_mask(legal_specs)
        return _ObservationView(vector=vector, legal_mask=legal_mask)

    # ------------------------------------------------------------------ legal action builders

    def _legal_action_specs(self, hand: HandEngine, player: int) -> Iterable[ActionSpec]:
        if hand.phase == HandPhase.AUCTION:
            yield from self._auction_actions(hand, player)
        elif hand.phase == HandPhase.MUSIK:
            yield from self._musik_actions(hand, player)
        elif hand.phase == HandPhase.PLAY:
            yield from self._play_actions(hand, player)

    def _auction_actions(self, hand: HandEngine, player: int) -> Iterable[ActionSpec]:
        if hand.auction.current_player != player:
            return []
        legal: List[ActionSpec] = []
        highest = hand.auction.highest_bid or 0
        for bid in BID_VALUES:
            if bid > highest and self._proof_for_bid(hand, player, bid, require_available=True):
                legal.append(ActionSpec(ActionType.BID, (bid,)))
        if hand.auction.highest_bid is not None and hand.auction.highest_bidder != player:
            legal.append(ActionSpec(ActionType.PASS, ()))
        return legal

    def _musik_actions(self, hand: HandEngine, player: int) -> Iterable[ActionSpec]:
        legal: List[ActionSpec] = []
        if hand.playing_player != player:
            return legal
        if hand.musik.chosen_index is None:
            legal.extend(ActionSpec(ActionType.CHOOSE_MUSIK, (idx,)) for idx in (0, 1))
        elif len(hand.hands[player]) > 10:
            legal.extend(self._card_pair_specs(hand, player))
        else:
            minimum = hand.winning_bid or 100
            default_contract = max(minimum, ((minimum + 9) // 10) * 10)
            legal.append(ActionSpec(ActionType.SET_CONTRACT, (default_contract,)))
        return legal

    def _play_actions(self, hand: HandEngine, player: int) -> Iterable[ActionSpec]:
        if hand.state is None or hand.state.current_player != player:
            return []
        legal_cards = hand.state.available_moves(player)
        return [
            ActionSpec(ActionType.PLAY_CARD, (self.catalog.encode_card(card),))
            for card in legal_cards
        ]

    # ------------------------------------------------------------------ helpers

    def _card_pair_specs(self, hand: HandEngine, player: int) -> Iterable[ActionSpec]:
        cards = hand.hands[player]
        if hand.state is not None:
            cards = hand.state.hands[player]
        encoded = sorted((self.catalog.encode_card(card), card) for card in cards)
        specs: List[ActionSpec] = []
        for i in range(len(encoded)):
            for j in range(i + 1, len(encoded)):
                pair_ids = (encoded[i][0], encoded[j][0])
                specs.append(ActionSpec(ActionType.RETURN_CARDS, pair_ids))
        return specs

    def _decode_card(self, hand: HandEngine, card_id: int) -> Card:
        sources: Sequence[Card]
        if hand.state is not None:
            sources = hand.state.hands[self.player]
        else:
            sources = hand.hands[self.player]
        for card in sources:
            if self.catalog.encode_card(card) == card_id:
                return card
        raise ValueError(f"Card id {card_id} not found in hand.")

    def _decode_card_pair(self, hand: HandEngine, pair: Tuple[int, int]) -> List[Card]:
        return [self._decode_card(hand, idx) for idx in pair]

    def _auto_meld_partner(self, hand: HandEngine, card: Card) -> Optional[Card]:
        if hand.state is None:
            return None
        state = hand.state
        if not state.current_trick.is_empty():
            return None
        if card.rank not in (Rank.KING, Rank.QUEEN):
            return None
        if card.suit in state.melds_declared[self.player]:
            return None
        partner_rank = Rank.QUEEN if card.rank is Rank.KING else Rank.KING
        for candidate in state.hands[self.player]:
            if candidate is card:
                continue
            if candidate.suit is card.suit and candidate.rank is partner_rank:
                return candidate
        return None

    def _remaining_cards(self, hand: HandEngine, player: int, opponent: int) -> Tuple[int, int]:
        if hand.state:
            return (
                len(hand.state.hands[player]),
                len(hand.state.hands[opponent]),
            )
        return (
            len(hand.hands[player]),
            len(hand.hands[opponent]),
        )

    def _encode_auction_history(
        self,
        hand: HandEngine,
        player: int,
        opponent: int,
        limit: int,
    ) -> List[float]:
        history = hand.auction.history[-limit:]
        encoded: List[float] = []
        for entry_player, action, amount in history:
            encoded.extend(
                [
                    1.0 if entry_player == player else 0.0,
                    1.0 if entry_player == opponent else 0.0,
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

    def _proof_for_bid(
        self,
        hand: HandEngine,
        player: int,
        bid: int,
        require_available: bool = False,
    ):
        required = bid - BASE_BID
        if required <= 0:
            return True if require_available else None
        cards = list(hand.hands[player]) if hand.state is None else list(hand.state.hands[player])
        for suit, points in MARRIAGE_POINTS.items():
            if points != required:
                continue
            king = next((card for card in cards if card.suit is suit and card.rank is Rank.KING), None)
            queen = next((card for card in cards if card.suit is suit and card.rank is Rank.QUEEN), None)
            if king and queen:
                proof = {"suit": suit, "cards": {king, queen}}
                return proof if not require_available else True
        return False if require_available else None
