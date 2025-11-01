"""Streamlit UI scaffold for Alpha-1000."""

from __future__ import annotations

import streamlit as st

from engine.cards import MARRIAGE_POINTS, Rank, Suit, card_label, deserialize_card
from engine.bidding import BASE_BID
from engine.service import HandService


def get_service() -> HandService:
    if "hand_service" not in st.session_state:
        st.session_state["hand_service"] = HandService()
    return st.session_state["hand_service"]


def rerun() -> None:
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def ensure_hand(service: HandService, starting_player: int = 0) -> None:
    service.start_new_hand(starting_player=starting_player)
    rerun()


def auto_proof(cards_payload: list[dict], bid_value: int):
    required = bid_value - BASE_BID
    if required <= 0:
        return None
    cards = [deserialize_card(payload) for payload in cards_payload]
    candidate = None
    for suit, points in MARRIAGE_POINTS.items():
        if points < required:
            continue
        king = next((c for c in cards if c.suit is suit and c.rank is Rank.KING), None)
        queen = next((c for c in cards if c.suit is suit and c.rank is Rank.QUEEN), None)
        if king and queen:
            candidate = {"suit": suit, "cards": {king, queen}}
            break
    return candidate


def unique_options(cards_payload: list[dict]) -> dict[str, dict]:
    options = {}
    for idx, payload in enumerate(cards_payload):
        label = card_label(deserialize_card(payload))
        key = f"{label} [{idx}]"
        options[key] = payload
    return options


def render_auction_controls(service: HandService, view) -> None:
    st.subheader("Auction")
    highest = view.highest_bid or 0
    st.write(f"Highest bid: {highest}")
    bid_value = st.number_input("Bid amount", min_value=100, max_value=200, step=10, value=max(100, highest + 10))
    if st.button("Place Bid"):
        proof = None
        if bid_value > 120:
            proof = auto_proof(view.hand, bid_value)
        try:
            service.place_bid(0, int(bid_value), proof=proof)
            rerun()
        except Exception as exc:
            st.error(f"Bid failed: {exc}")
    if st.button("Pass"):
        try:
            service.pass_bid(0)
            rerun()
        except Exception as exc:
            st.error(f"Cannot pass: {exc}")


def render_musik_controls(service: HandService, view) -> None:
    st.subheader("Musik")
    if view.musik["chosen_index"] is None:
        cols = st.columns(2)
        if cols[0].button("Take musik 0"):
            try:
                service.choose_musik(0, index=0)
                rerun()
            except Exception as exc:
                st.error(str(exc))
        if cols[1].button("Take musik 1"):
            try:
                service.choose_musik(0, index=1)
                rerun()
            except Exception as exc:
                st.error(str(exc))
    else:
        st.write("Select two cards to return to the unused musik.")
        options = unique_options(view.hand)
        selection = st.multiselect("Cards to return", options.keys())
        if st.button("Return selected cards"):
            if len(selection) != 2:
                st.warning("Select exactly two cards.")
            else:
                try:
                    cards = [options[key] for key in selection]
                    service.return_musik_cards(0, cards)
                    rerun()
                except Exception as exc:
                    st.error(str(exc))


def render_play_controls(service: HandService, view) -> None:
    st.subheader("Play")
    if view.trick:
        st.write("Current trick:")
        for play in view.trick.plays:
            st.write(f"Player {play.player}: {play.label}")

    if view.current_player != 0:
        st.info(f"Waiting for player {view.current_player}.")
        return

    if not view.legal_moves:
        st.warning("No legal moves available.")
        return

    options = unique_options(view.legal_moves)
    selection = st.selectbox("Play a card", list(options.keys()))
    declare = st.checkbox("Declare meld", value=False)
    partner_selection = None
    partner_payload = None

    if declare:
        hand_options = unique_options(view.hand)
        partner_selection = st.selectbox("Meld partner", list(hand_options.keys()))
        if partner_selection:
            partner_payload = hand_options[partner_selection]

    if st.button("Play selected card"):
        if declare and partner_payload is None:
            st.warning("Select a meld partner when declaring a meld.")
            return
        if declare and partner_payload == options[selection]:
            st.warning("Meld partner must be different from the card being played.")
            return
        try:
            service.play_card(0, options[selection], declare_meld=declare, meld_partner=partner_payload)
            rerun()
        except Exception as exc:
            st.error(str(exc))


def render_complete_controls(service: HandService) -> None:
    st.subheader("Hand complete")
    if st.button("Finalize scoring"):
        try:
            service.finish_hand()
            rerun()
        except Exception as exc:
            st.error(str(exc))


def render_status(view) -> None:
    st.write(f"Phase: {view.phase}")
    if view.winning_bid:
        st.write(f"Winning bid: {view.winning_bid}")
    if view.contract:
        st.write(f"Contract: {view.contract}")
    if view.trump:
        st.write(f"Trump suit: {view.trump}")

    st.write(f"Meld points: {view.meld_points}")
    st.write(f"Card points: {view.card_points}")
    st.write(f"Hand sizes: {view.remaining_cards}")

    with st.expander("Auction history"):
        for entry in view.auction_history:
            action = entry["action"]
            if action == "bid":
                st.write(f"Player {entry['player']} bid {entry['amount']}")
            else:
                st.write(f"Player {entry['player']} passed")


def main() -> None:
    st.set_page_config(page_title="Alpha-1000 Sandbox", layout="wide")
    st.title("Alpha-1000")

    service = get_service()
    session_view = service.get_session_view(0)

    st.sidebar.header("Session Controls")
    st.sidebar.write(f"Scores: {session_view.scores}")
    if st.sidebar.button("Start new hand (you lead)"):
        ensure_hand(service, starting_player=0)
    if st.sidebar.button("Start new hand (opponent leads)"):
        ensure_hand(service, starting_player=1)

    if session_view.hand is None:
        st.info("Start a new hand to begin.")
        return

    view = session_view.hand
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Your hand")
        for label in [card_label(deserialize_card(payload)) for payload in view.hand]:
            st.write(label)
    with cols[1]:
        render_status(view)

    if view.phase == "auction" and view.current_player == 0:
        render_auction_controls(service, view)
    elif view.phase == "musik" and view.playing_player == 0:
        render_musik_controls(service, view)
    elif view.phase == "play":
        render_play_controls(service, view)
    elif view.phase == "complete":
        render_complete_controls(service)


if __name__ == "__main__":
    main()
