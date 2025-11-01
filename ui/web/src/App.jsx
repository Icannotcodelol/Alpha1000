import { useEffect, useMemo, useState } from 'react';

// Use relative URLs since the API is served from the same domain
const API_BASE = import.meta.env.VITE_API_BASE ?? '';

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      'ngrok-skip-browser-warning': 'true',
      ...options.headers,
    },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || response.statusText);
  }
  return response.json();
}

const SUIT_SYMBOL = {
  Hearts: '♥',
  Diamonds: '♦',
  Clubs: '♣',
  Spades: '♠',
};

const SUIT_CLASS = {
  Hearts: 'card--red',
  Diamonds: 'card--red',
  Clubs: 'card--black',
  Spades: 'card--black',
};

function Card({ card }) {
  if (!card || !card.suit) {
    return <span className="card card--unknown">?</span>;
  }
  const suit = card.suit;
  return (
    <span className={`card ${SUIT_CLASS[suit] ?? ''}`}>
      <span className="card__rank">{card.rank}</span>
      <span className="card__suit">{SUIT_SYMBOL[suit] ?? suit[0]}</span>
    </span>
  );
}

function ActionCard({ action, onSelect, disabled }) {
  const label = action.label.replace(/^Play /, '');
  const [rank, , suitName] = label.split(' ');
  return (
    <button
      className={`card card--play ${SUIT_CLASS[suitName] ?? ''}`}
      disabled={disabled}
      onClick={() => onSelect(action.action_id)}
    >
      <span className="card__rank">{rank}</span>
      <span className="card__suit">{SUIT_SYMBOL[suitName] ?? suitName?.[0]}</span>
    </button>
  );
}

function ActionPill({ action, onSelect, disabled }) {
  return (
    <button className="pill" onClick={() => onSelect(action.action_id)} disabled={disabled}>
      {action.label}
    </button>
  );
}

export default function App() {
  const [sessionId, setSessionId] = useState(null);
  const [state, setState] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const startSession = async () => {
    try {
      setLoading(true);
      console.log('Starting session with API_BASE:', API_BASE);
      const payload = await api('/session/start', {
        method: 'POST',
        body: JSON.stringify({}),
      });
      console.log('Session started:', payload);
      setSessionId(payload.session_id);
      setState(payload.state);
      setError(null);
    } catch (err) {
      console.error('Failed to start session:', err);
      setError('Load failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    startSession();
  }, []);

  const handleAction = async (actionId) => {
    if (!sessionId) return;
    try {
      setLoading(true);
      const payload = await api(`/session/${sessionId}/action`, {
        method: 'POST',
        body: JSON.stringify({ action_id: actionId }),
      });
      setState(payload.state);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const resetMatch = async () => {
    if (!sessionId) return;
    try {
      setLoading(true);
      const payload = await api(`/session/${sessionId}/reset`, {
        method: 'POST',
        body: JSON.stringify({}),
      });
      setState(payload.state);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const formattedActions = useMemo(() => {
    if (!state?.legalActions) return { cardActions: [], metaActions: [] };
    const cardActions = [];
    const metaActions = [];
    for (const action of state.legalActions) {
      if (action.label.startsWith('Play ')) {
        cardActions.push(action);
      } else {
        metaActions.push(action);
      }
    }
    return { cardActions, metaActions };
  }, [state]);

  if (!state) {
    return (
      <div className="app">
        <div className="table">
          <div className="table__header">
            <h1>Alpha1000 Arena</h1>
          </div>
          {error && (
            <div className="error" style={{ padding: '20px', margin: '20px', background: '#ff4444', color: 'white', borderRadius: '8px' }}>
              <strong>Error:</strong> {error}
              <br />
              <button onClick={startSession} style={{ marginTop: '10px', padding: '10px', cursor: 'pointer' }}>
                Retry
              </button>
            </div>
          )}
          <p className="loading">Initialising match...</p>
        </div>
      </div>
    );
  }

  const { scores, hand, currentTrick, auction, handComplete, handSummary, matchComplete, opponentName } = state;

  return (
    <div className="app">
      <div className="table">
        <div className="table__header">
          <div>
            <h1>Alpha1000 Arena</h1>
            <p className="opponent">Opponent: {opponentName}</p>
          </div>
          <div className="scoreboard">
            <div className="scoreboard__label">Score</div>
            <div className="scoreboard__values">
              <div className="scoreboard__value">
                <span>You</span>
                <strong>{scores.agent}</strong>
              </div>
              <div className="scoreboard__value">
                <span>Champion</span>
                <strong>{scores.opponent}</strong>
              </div>
            </div>
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        <div className="table__body">
          <aside className="sidebar">
            <section className="panel">
              <h3>Phase</h3>
              <p className="phase-tag">{state.phase}</p>
            </section>

            <section className="panel">
              <h3>Auction</h3>
              <p className="panel__line">
                Highest bid: <strong>{auction.highestBid ?? '—'}</strong>
              </p>
              <p className="panel__line">
                Leader: <strong>{auction.highestBidder ?? '—'}</strong>
              </p>
              {auction.history.length > 0 && (
                <ul className="history">
                  {auction.history.slice(-6).map((item, idx) => (
                    <li key={idx}>
                      P{item.player}: {item.action}{item.amount ? ` ${item.amount}` : ''}
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {handComplete && handSummary && (
              <section className="panel panel--highlight">
                <h3>Hand Summary</h3>
                <p>Contract success: <strong>{String(handSummary.contractSuccess)}</strong></p>
                {handSummary.episodeStats && (
                  <pre>{JSON.stringify(handSummary.episodeStats, null, 2)}</pre>
                )}
              </section>
            )}

            {matchComplete && (
              <section className="panel panel--highlight">
                <h3>Match Complete</h3>
                <button className="pill pill--wide" onClick={resetMatch} disabled={loading}>
                  Play Again
                </button>
              </section>
            )}
          </aside>

          <main className="play-area">
            <section className="trick-area">
              <h3>Table</h3>
              <div className="trick-cards">
                {currentTrick.length === 0 ? (
                  <div className="placeholder">No cards on table.</div>
                ) : (
                  currentTrick.map((play, idx) => (
                    <div key={idx} className="trick-card">
                      <span className="trick-card__player">Player {play.player}</span>
                      <Card card={play.card} />
                    </div>
                  ))
                )}
              </div>
            </section>

            <section className="hand-area">
              <h3>Your Hand</h3>
              <div className="hand">
                {hand.map((card, idx) => (
                  <Card key={`${card.suit}-${card.rank}-${idx}`} card={card} />
                ))}
              </div>
            </section>

            {!matchComplete && (
              <section className="actions-area">
                <h3>Your Move</h3>
                {formattedActions.metaActions.length > 0 && (
                  <div className="action-group">
                    <h4>Actions</h4>
                    <div className="action-buttons">
                      {formattedActions.metaActions.map((action) => (
                        <ActionPill
                          key={action.action_id}
                          action={action}
                          onSelect={handleAction}
                          disabled={loading}
                        />
                      ))}
                    </div>
                  </div>
                )}
                <div className="action-group">
                  <h4>Playable Cards</h4>
                  {formattedActions.cardActions.length === 0 ? (
                    <div className="placeholder">Waiting for opponent...</div>
                  ) : (
                    <div className="card-actions">
                      {formattedActions.cardActions.map((action) => (
                        <ActionCard
                          key={action.action_id}
                          action={action}
                          onSelect={handleAction}
                          disabled={loading}
                        />
                      ))}
                    </div>
                  )}
                </div>
              </section>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
