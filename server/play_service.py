"""REST service to play Tysiąc against a champion policy."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from bots.self_play import SelfPlayBot
from engine.cards import Card
from engine.game import HandPhase
from rl.action import ActionCatalog, ActionSpec, ActionType
from rl.env import Observation, TysiacEnv


DEFAULT_CHAMPION = Path("data/checkpoints/ppo_update_6070.pt")
STATIC_DIR = Path(__file__).parent.parent / "ui" / "web" / "dist"


class StartRequest(BaseModel):
    opponent_checkpoint: Optional[str] = None
    target_score: int = 1000


class ActionRequest(BaseModel):
    action_id: int


class ResetRequest(BaseModel):
    target_score: Optional[int] = None


class SessionState:
    def __init__(self, env: TysiacEnv, catalog: ActionCatalog, obs: Observation, target_score: int) -> None:
        self.env = env
        self.catalog = catalog
        self.obs = obs
        self.target_score = target_score
        self.opponent_name = env.opponent_name


sessions: Dict[str, SessionState] = {}


app = FastAPI(title="Alpha1000 Play Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_opponent(checkpoint_path: Path) -> SelfPlayBot:
    if not checkpoint_path.exists():
        raise HTTPException(status_code=404, detail=f"Checkpoint {checkpoint_path} not found")
    return SelfPlayBot(checkpoint_path, device="cpu")


def agent_hand(env: TysiacEnv) -> List[Card]:
    if env.hand is None:
        return []
    if env.hand.state is not None:
        return list(env.hand.state.hands[env.agent_player])
    return list(env.hand.hands[env.agent_player])


def describe_card(card: Optional[Card]) -> Dict[str, Optional[str]]:
    if card is None:
        return {"suit": None, "rank": None}
    return {"suit": card.suit.name.title(), "rank": card.rank.name.title()}


def card_from_id(cards: List[Card], catalog: ActionCatalog, card_id: int) -> Optional[Card]:
    for card in cards:
        if catalog.encode_card(card) == card_id:
            return card
    return None


def describe_action(env: TysiacEnv, spec: ActionSpec, catalog: ActionCatalog) -> str:
    hand_cards = agent_hand(env)
    if spec.action_type == ActionType.BID:
        return f"Bid {spec.payload[0]}"
    if spec.action_type == ActionType.PASS:
        return "Pass"
    if spec.action_type == ActionType.CHOOSE_MUSIK:
        return f"Choose musik {spec.payload[0]}"
    if spec.action_type == ActionType.RETURN_CARDS:
        cards = [card_from_id(hand_cards, catalog, cid) for cid in spec.payload]
        readable = ", ".join(
            f"{card.rank.name.title()} of {card.suit.name.title()}" if card else f"Card#{cid}"
            for cid, card in zip(spec.payload, cards)
        )
        return f"Return {readable}"
    if spec.action_type == ActionType.SET_CONTRACT:
        return f"Set contract {spec.payload[0]}"
    if spec.action_type == ActionType.PLAY_CARD:
        card = card_from_id(hand_cards, catalog, spec.payload[0])
        return f"Play {card.rank.name.title()} of {card.suit.name.title()}" if card else f"Play card #{spec.payload[0]}"
    return spec.action_type.name


def list_legal_actions(env: TysiacEnv, obs: Observation, catalog: ActionCatalog) -> List[Dict[str, object]]:
    actions: List[Dict[str, object]] = []
    for action_id, allowed in enumerate(obs.legal_mask):
        if allowed:
            spec = catalog.spec(action_id)
            actions.append({
                "action_id": action_id,
                "label": describe_action(env, spec, catalog),
            })
    return actions


def serialize_state(session: SessionState, obs: Observation, *, hand_summary: Optional[Dict[str, object]] = None, match_complete: bool = False) -> Dict[str, object]:
    env = session.env
    catalog = session.catalog
    state: Dict[str, object] = {
        "phase": obs.info.get("phase"),
        "scores": {
            "agent": env.session.scores[env.agent_player],
            "opponent": env.session.scores[env.opponent_player],
        },
        "opponentName": session.opponent_name,
        "legalActions": [] if match_complete else list_legal_actions(env, obs, catalog),
        "hand": [describe_card(card) for card in agent_hand(env)],
        "currentTrick": [
            {
                "player": play[0],
                "card": describe_card(play[1]),
            }
            for play in (env.hand.state.current_trick.plays if env.hand and env.hand.state else [])
        ],
        "auction": {
            "highestBid": env.hand.auction.highest_bid if env.hand else None,
            "highestBidder": env.hand.auction.highest_bidder if env.hand else None,
            "history": [
                {
                    "player": player,
                    "action": action,
                    "amount": amount,
                }
                for (player, action, amount) in (env.hand.auction.history if env.hand else [])
            ],
        },
        "handComplete": hand_summary is not None,
        "handSummary": hand_summary,
        "matchComplete": match_complete,
    }
    return state


def ensure_session(session_id: str) -> SessionState:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/session/start")
def start_session(request: StartRequest) -> Dict[str, object]:
    checkpoint_path = Path(request.opponent_checkpoint) if request.opponent_checkpoint else DEFAULT_CHAMPION
    opponent = load_opponent(checkpoint_path)
    env = TysiacEnv(opponent=opponent, agent_player=0, include_seen_cards=True)
    catalog = env.catalog
    obs = env.reset()
    session = SessionState(env=env, catalog=catalog, obs=obs, target_score=request.target_score)
    session_id = uuid.uuid4().hex
    sessions[session_id] = session
    return {
        "session_id": session_id,
        "state": serialize_state(session, obs),
    }


@app.post("/session/{session_id}/action")
def take_action(session_id: str, request: ActionRequest) -> Dict[str, object]:
    session = ensure_session(session_id)
    env = session.env
    if request.action_id < 0 or request.action_id >= len(session.obs.legal_mask) or not session.obs.legal_mask[request.action_id]:
        raise HTTPException(status_code=400, detail="Illegal action")
    obs, reward, done, info = env.step(request.action_id)
    session.obs = obs
    hand_summary: Optional[Dict[str, object]] = None
    match_complete = False
    if done:
        hand_summary = {
            "contractSuccess": info.get("contract_success"),
            "episodeStats": info.get("episode_stats", {}),
        }
        match_complete = max(env.session.scores) >= session.target_score
        if not match_complete:
            session.obs = env.reset()
            obs = session.obs
    return {
        "state": serialize_state(session, session.obs, hand_summary=hand_summary, match_complete=match_complete),
        "reward": reward,
    }


@app.post("/session/{session_id}/reset")
def reset_session(session_id: str, request: ResetRequest) -> Dict[str, object]:
    session = ensure_session(session_id)
    if request.target_score is not None:
        session.target_score = request.target_score
    session.env.session.scores = [0, 0]
    session.obs = session.env.reset()
    return {"state": serialize_state(session, session.obs)}


# Mount static files and serve index.html for the React app
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")
    
    @app.get("/")
    def serve_index():
        return FileResponse(STATIC_DIR / "index.html")
