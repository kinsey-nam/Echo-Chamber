import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import textwrap
import os

# ----------------------------------------
# AI helper (replace with Gemini/Claude/etc.)
# ----------------------------------------
AI_API_KEY = os.getenv("AI_API_KEY")
AI_API_URL = os.getenv("AI_API_URL", "https://api.openai.com/v1/chat/completions")
AI_MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")

def call_ai(system_prompt, user_prompt, max_tokens=600):
    if not AI_API_KEY:
        return "AI key not configured. Please set AI_API_KEY environment variable."
    headers = {
        "Authorization": f"Bearer {AI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": AI_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
    }
    resp = requests.post(AI_API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()

# ----------------------------------------
# Markov chain utilities
# ----------------------------------------
STATE_NAMES = ["Discovery", "Rabbit Hole", "Echo Chamber"]

def is_valid_prob_vector(vec, tol=1e-6):
    return np.all(vec >= 0) and np.all(vec <= 1) and abs(np.sum(vec) - 1.0) <= tol

def is_valid_transition_matrix(P, tol=1e-6):
    if P.shape != (3, 3):
        return False
    if np.any(P < 0) or np.any(P > 1):
        return False
    row_sums = P.sum(axis=1)
    return np.all(np.abs(row_sums - 1.0) <= tol)

def simulate_markov_chain(P, v0, T):
    """Return an array of shape (T+1, 3) with probability vectors over time."""
    probs = np.zeros((T + 1, 3))
    probs[0, :] = v0
    for t in range(1, T + 1):
        probs[t, :] = probs[t - 1, :].dot(P)
    return probs

def matrix_power(P, n):
    return np.linalg.matrix_power(P, n)

def is_irreducible(P):
    A = (P > 0).astype(int)
    reach = np.copy(A)
    for _ in range(3):  # up to number of states
        reach = reach + reach.dot(A)
    return np.all(reach > 0)

def gcd(numbers):
    from math import gcd as _gcd
    from functools import reduce
    return reduce(_gcd, numbers)

def is_aperiodic(P, max_steps=20, tol=1e-12):
    n_states = P.shape[0]
    Pn = np.eye(n_states)
    flags = []
    for i in range(n_states):
        return_times = []
        Pn = np.eye(n_states)
        for n in range(1, max_steps + 1):
            Pn = Pn.dot(P)
            if Pn[i, i] > tol:
                return_times.append(n)
        if not return_times:
            flags.append(False)
            continue
        period_i = gcd(return_times)
        flags.append(period_i == 1)
    return all(flags)

# ----------------------------------------
# Streamlit layout
# ----------------------------------------
st.set_page_config(
    page_title="Social Media Echo Chamber Markov Chain",
    layout="wide"
)

st.title("Social Media Echo Chamber: 3-State Markov Chain Simulator")

st.markdown(
    "States: **Discovery** (diverse content), **Rabbit Hole** (increasingly similar content), "
    "**Echo Chamber** (highly homogeneous content)."
)

# Sidebar controls
st.sidebar.header("Simulation Controls")

T = st.sidebar.number_input(
    "Number of time steps",
    min_value=1,
    max_value=500,
    value=50,
    step=1,
)

n_sims = st.sidebar.number_input(
    "Number of simulations (informational; model is analytic)",
    min_value=1,
    max_value=1000,
    value=1,
    step=1,
)

st.sidebar.caption(
    "Probabilities are computed exactly via \( v_0 P^n \), "
    "not by Monte Carlo sampling."
)

# ----------------------------------------
# Input section
# ----------------------------------------
st.header("Input: Initial State and Transition Matrix")

col_v, col_P = st.columns([1, 3])

with col_v:
    st.subheader("Initial state vector \(v_0\)")
    st.markdown("Specify the starting probability for each state (must sum to 1).")

    v0_d = st.number_input("Discovery (v0[0])", 0.0, 1.0, 1.0, 0.01)
    v0_r = st.number_input("Rabbit Hole (v0[1])", 0.0, 1.0, 0.0, 0.01)
    v0_e = st.number_input("Echo Chamber (v0[2])", 0.0, 1.0, 0.0, 0.01)
    v0 = np.array([v0_d, v0_r, v0_e], dtype=float)

with col_P:
    st.subheader("Transition matrix \(P\)")
    st.markdown(
        "Each row is the probability of moving **from** one state **to** the next state "
        "in a single time step (rows must each sum to 1)."
    )

    P = np.zeros((3, 3))
    labels = ["From Discovery", "From Rabbit Hole", "From Echo Chamber"]
    defaults = [
        [0.3, 0.5, 0.2],
        [0.1, 0.7, 0.2],
        [0.05, 0.2, 0.75],
    ]

    for i in range(3):
        st.markdown(f"**{labels[i]}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            P[i, 0] = st.number_input(
                f"→ Discovery (row {i+1}, col 1)",
                0.0, 1.0, float(defaults[i][0]), 0.01
            )
        with c2:
            P[i, 1] = st.number_input(
                f"→ Rabbit Hole (row {i+1}, col 2)",
                0.0, 1.0, float(defaults[i][1]), 0.01
            )
        with c3:
            P[i, 2] = st.number_input(
                f"→ Echo Chamber (row {i+1}, col 3)",
                0.0, 1.0, float(defaults[i][2]), 0.01
            )

# Validation
valid_v0 = is_valid_prob_vector(v0)
valid_P = is_valid_transition_matrix(P)

if not valid_v0:
    st.error(f"Initial state vector v0 must be in [0,1] and sum to 1. Current sum = {v0.sum():.4f}")
if not valid_P:
    st.error(
        "Each row of P must have entries in [0,1] and row sums must be 1.\n"
        f"Current row sums: {[f'{s:.4f}' for s in P.sum(axis=1)]}"
    )

run = st.button("Run Simulation", disabled=not (valid_v0 and valid_P))

# ----------------------------------------
# Results section
# ----------------------------------------
st.header("Results")

if run:
    probs = simulate_markov_chain(P, v0, T)
    time_steps = np.arange(T + 1)

    df_probs = pd.DataFrame(
        probs,
        columns=STATE_NAMES,
        index=time_steps
    )
    df_probs.index.name = "Time step"

    # Line chart + metrics
    line_col, metrics_col = st.columns([3, 1])

    with line_col:
        st.subheader("State probabilities over time")
        st.line_chart(df_probs)  # one line per state[web:37]
    with metrics_col:
        st.subheader("Final-step probabilities")
        final_probs = df_probs.iloc[-1]
        st.metric("P(Discovery)", f"{final_probs['Discovery']:.3f}")
        st.metric("P(Rabbit Hole)", f"{final_probs['Rabbit Hole']:.3f}")
        st.metric("P(Echo Chamber)", f"{final_probs['Echo Chamber']:.3f}")

    # Slider: probabilities at a chosen time step n
    st.subheader("Inspect probabilities at a chosen time step")

    n_view = st.slider("Select time step n", 0, T, min(10, T))  # main slider[web:56]
    probs_n = df_probs.loc[n_view]

    bar_col, table_col = st.columns([2, 2])

    with bar_col:
        st.markdown(f"**Probability distribution at time step n = {n_view}**")
        st.bar_chart(
            pd.DataFrame(
                probs_n,
                columns=["Probability"]
            )
        )  # bar chart for that n[web:35]
    with table_col:
        st.markdown("Numeric values at this n")
        st.dataframe(
            probs_n.to_frame(name="Probability").style.format("{:.4f}")
        )

    # Heatmap over time
    st.subheader("Heatmap: probability of each state over time")
    fig_hm, ax_hm = plt.subplots(figsize=(6, 3))
    im = ax_hm.imshow(probs.T, aspect="auto", origin="lower", cmap="viridis")
    ax_hm.set_yticks(range(len(STATE_NAMES)))
    ax_hm.set_yticklabels(STATE_NAMES)
    ax_hm.set_xlabel("Time step")
    ax_hm.set_title("Heatmap of state probabilities")
    fig_hm.colorbar(im, ax=ax_hm, label="Probability")
    fig_hm.tight_layout()
    st.pyplot(fig_hm)

    # P^n display
    st.subheader("Transition matrix \(P^n\)")
    n_for_matrix = st.number_input(
        "Time step n for P^n", min_value=1, max_value=500, value=min(10, T), step=1
    )
    Pn = matrix_power(P, int(n_for_matrix))
    df_Pn = pd.DataFrame(Pn, columns=STATE_NAMES, index=STATE_NAMES)
    st.markdown(f"Matrix \(P^{int(n_for_matrix)}\):")
    st.dataframe(df_Pn.style.format("{:.4f}"))

    # Chain properties
    st.subheader("Chain properties")

    irreducible = is_irreducible(P)
    aperiodic = is_aperiodic(P)

    col_ir, col_ap = st.columns(2)
    with col_ir:
        st.metric("Irreducible?", "Yes" if irreducible else "No")
    with col_ap:
        st.metric("Aperiodic?", "Yes" if aperiodic else "No")

    st.markdown(
        "- **Irreducible**: every state can eventually reach every other state.\n"
        "- **Aperiodic**: returns to a state do not happen on a fixed cycle only."
    )

    # ----------------------------------------
    # AI: state transition diagram
    # ----------------------------------------
    st.subheader("AI-generated State Transition Diagram (ASCII)")

    diagram_prompt = f"""
    Create a clear ASCII/Markdown state transition diagram for a 3-state Markov Chain
    modelling a social media echo chamber with states:
    1. Discovery
    2. Rabbit Hole
    3. Echo Chamber

    Use the following transition matrix P (rows = from, cols = to). Only draw arrows for
    transitions with probability > 0.01, and annotate each arrow with the probability
    (rounded to 2 decimals):

    P =
    {np.array2string(P, precision=3)}

    Use a simple text diagram, for example:

    Discovery --> Rabbit Hole (0.40)
    Discovery --> Echo Chamber (0.10)
    ...

    Make it compact and readable.
    """

    if st.button("Ask AI for transition diagram"):
        diagram = call_ai(
            "You are an assistant that outputs only plain text diagrams, no explanations.",
            diagram_prompt,
        )
        st.text(diagram)

    # ----------------------------------------
    # AI: explanation of irreducible/aperiodic
    # ----------------------------------------
    st.subheader("AI explanation of chain behaviour")

    explain_prompt = f"""
    You are explaining properties of a 3-state Markov Chain model for a social media echo chamber
    with states: Discovery, Rabbit Hole, Echo Chamber.

    The transition matrix is:
    {np.array2string(P, precision=3)}

    The chain has:
    - irreducible = {irreducible}
    - aperiodic = {aperiodic}

    Provide a concise explanation (max 150 words) of:
    - what irreducible and aperiodic mean in intuitive terms for this social-media context
    - whether this specific chain is irreducible and aperiodic, and what that implies about
      long-run behaviour (e.g., convergence to a stationary distribution).

    Use plain language suitable for non-experts; no formulas.
    """

    if st.button("Ask AI to explain chain properties"):
        explanation = call_ai(
            "You are a concise technical explainer. Answer in under 150 words.",
            explain_prompt,
        )
        st.markdown(textwrap.dedent(explanation))

else:
    st.info("Adjust the inputs, then click **Run Simulation** once all probabilities are valid.")

