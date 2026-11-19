"""
Module 5: Sequential Data Modelling (Unit 4).
Implements:
- Discrete Markov Chains & Transition Matrix Analysis
- Stationary / Steady-State Probability Distribution
- Hidden Markov Models (HMM) Parameterization
- Forward Algorithm (Evaluation Problem)
- Backward Algorithm (Evaluation Problem verification)
- Viterbi Algorithm (Decoding Problem - Optimal State Trajectory)
- State Posterior Probabilities
- Separate Sequential Dataset Uploader & Vehicle Maintenance Telemetry Demo
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.styling import render_section_title, render_metric_card
from utils.hmm_algorithms import MarkovModel, HiddenMarkovModel
from modules.data_loader import load_sequential_dataset

def render_sequential_models_page():
    """Renders Unit 4 Sequential and Temporal Data Modelling Suite."""
    st.markdown("""
    <div style="background: linear-gradient(135deg, #102A43 0%, #1E3E62 100%); padding: 20px; border-radius: 12px; border-left: 5px solid #008DDA; margin-bottom: 24px;">
        <h2 style="color: #FFFFFF; margin: 0 0 6px 0; font-size: 1.6rem;">⏱️ Unit 4: Sequential Data Modelling & Hidden Markov Models</h2>
        <p style="color: #A0C4E2; margin: 0; font-size: 0.95rem;">
            Discrete Markov chains, stationary distributions, and complete Hidden Markov Model inference: Forward, Backward, and Viterbi algorithms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Academic & Engineering Integrity Disclaimer
    st.warning("""
    ⚠️ **Methodology Note on Sequential Modelling**:
    Standard used-car tabular datasets consist of cross-sectional, independent and identically distributed (i.i.d.) vehicle listings. Tabular car listings do not represent a single time-ordered sequential process. Imposing an HMM on unordered listings would violate the fundamental **Markov Property** ($P(S_t \mid S_{t-1}, \dots, S_0) = P(S_t \mid S_{t-1})$).
    
    To ensure genuine mathematical validity for your Unit 4 project review, this module provides:
    1. A **Dedicated Sequential Telemetry Suite** simulating continuous vehicle lifecycle degradation states (Normal $\\to$ Wear $\\to$ Critical Fault) with observable diagnostic symptoms.
    2. An option below to **Upload Any Custom Sequential CSV** (e.g., sensor telemetry, economic regimes, stock ticks).
    """)

    # Sequential Dataset Loader
    with st.expander("📂 Sequential Dataset Management & Upload Option", expanded=False):
        uploaded_seq = st.file_uploader("Upload a Custom Sequential CSV File (Optional)", type=["csv"], key="seq_uploader")
        if uploaded_seq is not None:
            seq_df = pd.read_csv(uploaded_seq)
            st.success(f"Custom sequential dataset loaded: {seq_df.shape[0]} rows, {seq_df.shape[1]} columns.")
        else:
            seq_df = load_sequential_dataset()
            st.info("Using built-in verified 'Vehicle Component Wear & Maintenance Sequence' dataset (800 sequential telemetry steps across 40 vehicles).")

        st.dataframe(seq_df.head(6), use_container_width=True)

    tab_markov, tab_hmm_arch, tab_forward_backward, tab_viterbi, tab_interactive = st.tabs([
        "⛓️ Discrete Markov Models",
        "🏗️ HMM Architecture & Parameters",
        "⏩ Forward & Backward Algorithms",
        "🧭 Viterbi Optimal Decoding",
        "🎮 Interactive Sequence Inference"
    ])

    # Standard Automotive Sequential Domain Setup
    states = ["Normal Operation", "Minor Wear", "Critical System Fault"]
    observations = ["Smooth Ride", "Engine Noise", "Warning Light", "Brake Squeal", "Performance Drop"]

    # Transition Matrix A (NxN)
    A_default = np.array([
        [0.85, 0.12, 0.03],
        [0.15, 0.70, 0.15],
        [0.40, 0.20, 0.40]
    ])

    # Emission Matrix B (NxM)
    B_default = np.array([
        [0.70, 0.10, 0.05, 0.10, 0.05],
        [0.20, 0.35, 0.15, 0.20, 0.10],
        [0.05, 0.20, 0.40, 0.10, 0.25]
    ])

    pi_default = np.array([0.70, 0.25, 0.05])

    # ==========================================
    # TAB 1: Markov Models
    # ==========================================
    with tab_markov:
        render_section_title("Discrete-Time First-Order Markov Chain")
        st.markdown(r"""
        A Markov chain satisfies the **Markovian Memoryless Property**:
        $$P(X_{t+1} = s_j \mid X_t = s_i, X_{t-1} = s_{t-1}, \dots, X_0 = s_0) = P(X_{t+1} = s_j \mid X_t = s_i) = A_{ij}$$
        """)

        mm = MarkovModel(states=states, transition_matrix=A_default, initial_probs=pi_default)

        m_col1, m_col2 = st.columns([1, 1])
        with m_col1:
            st.write("**One-Step State Transition Matrix (A):**")
            trans_df = pd.DataFrame(mm.A, index=states, columns=states)
            st.dataframe(trans_df.style.format("{:.2%}"), use_container_width=True)

        with m_col2:
            st.write("**Stationary / Steady-State Distribution ($\pi_{\infty}$):**")
            steady_state = mm.compute_steady_state()
            steady_df = pd.DataFrame({
                "Hidden State": states,
                "Initial Prior π₀": mm.pi,
                "Steady-State Probability π∞": steady_state
            })
            st.dataframe(steady_df.style.format({"Initial Prior π₀": "{:.2%}", "Steady-State Probability π∞": "{:.2%}"}), use_container_width=True)

        # Multi-Step Transition Power A^k
        render_section_title("Multi-Step Transition Analysis (A^k)")
        k_steps = st.slider("Select Horizon (k steps ahead):", 1, 20, 3)
        A_k = mm.get_k_step_transition(k_steps)
        st.write(f"**State Transition Probability Matrix After {k_steps} Maintenance Intervals (A^{k_steps}):**")
        st.dataframe(pd.DataFrame(A_k, index=states, columns=states).style.format("{:.2%}"), use_container_width=True)

        # Visual State Heatmap
        fig_trans = px.imshow(
            A_k, x=states, y=states, text_auto=".1%",
            title=f"Markov Transition Probabilities after {k_steps} intervals",
            color_continuous_scale="Blues", template="plotly_dark"
        )
        fig_trans.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
        st.plotly_chart(fig_trans, use_container_width=True)

    # ==========================================
    # TAB 2: HMM Architecture
    # ==========================================
    with tab_hmm_arch:
        render_section_title("Hidden Markov Model Parameters: λ = (A, B, π)")
        st.markdown(r"""
        In an HMM:
        - **Hidden States ($S$)**: Vehicle physical condition (cannot be directly observed without disassembling).
        - **Observable Symptoms ($V$)**: Telemetry indicators (diagnostic lights, noises, vibration).
        - **Transition Probability Matrix ($A$)**: $A_{ij} = P(q_{t+1} = S_j \mid q_t = S_i)$
        - **Emission Probability Matrix ($B$)**: $B_j(k) = P(O_t = v_k \mid q_t = S_j)$
        - **Initial State Distribution ($\pi$)**: $\pi_i = P(q_1 = S_i)$
        """)

        h_col1, h_col2 = st.columns([1, 1])
        with h_col1:
            st.write("**Hidden State Transition Matrix A:**")
            st.dataframe(pd.DataFrame(A_default, index=states, columns=states).style.format("{:.2%}"), use_container_width=True)

        with h_col2:
            st.write("**Observation Emission Matrix B:**")
            st.dataframe(pd.DataFrame(B_default, index=states, columns=observations).style.format("{:.2%}"), use_container_width=True)

        # Heatmap of Emissions
        fig_b = px.imshow(
            B_default, x=observations, y=states, text_auto=".1%",
            title="Emission Probabilities B_j(k): P(Symptom | Hidden Health State)",
            color_continuous_scale="Teal", template="plotly_dark"
        )
        fig_b.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
        st.plotly_chart(fig_b, use_container_width=True)

    # Instantiate HMM engine
    hmm = HiddenMarkovModel(
        states=states,
        observations=observations,
        transition_matrix=A_default,
        emission_matrix=B_default,
        initial_probs=pi_default
    )

    # Sample observation sequence for algorithmic walkthrough
    demo_seq = ["Smooth Ride", "Smooth Ride", "Engine Noise", "Warning Light", "Performance Drop"]

    # ==========================================
    # TAB 3: Forward & Backward Algorithms
    # ==========================================
    with tab_forward_backward:
        render_section_title("Problem 1: Evaluation Problem P(O | λ)")
        st.write(f"Demonstrating evaluation on sequence: **{demo_seq}**")

        f_prob, alpha_trellis = hmm.forward_algorithm(demo_seq)
        b_prob, beta_trellis = hmm.backward_algorithm(demo_seq)

        f_c1, f_c2 = st.columns(2)
        with f_c1:
            render_metric_card("Forward P(O | λ)", f"{f_prob:.5e}", "Computed via Forward Trellis")
        with f_c2:
            render_metric_card("Backward P(O | λ)", f"{b_prob:.5e}", "Exact Match Verification")

        st.markdown(r"""
        #### 1. Forward Recursion ($\alpha_t(i)$)
        - **Initialization**: $\alpha_1(i) = \pi_i B_i(O_1)$
        - **Induction**: $\alpha_{t+1}(j) = \left[ \sum_{i=1}^N \alpha_t(i) A_{ij} \right] B_j(O_{t+1})$
        - **Termination**: $P(O \mid \lambda) = \sum_{i=1}^N \alpha_T(i)$
        """)

        alpha_df = pd.DataFrame(alpha_trellis, columns=states, index=[f"t={t+1}: {demo_seq[t]}" for t in range(len(demo_seq))])
        st.dataframe(alpha_df.style.format("{:.5e}"), use_container_width=True)

        st.markdown(r"""
        #### 2. Backward Recursion ($\beta_t(i)$)
        - **Initialization**: $\beta_T(i) = 1.0$
        - **Induction**: $\beta_t(i) = \sum_{j=1}^N A_{ij} B_j(O_{t+1}) \beta_{t+1}(j)$
        - **Termination**: $P(O \mid \lambda) = \sum_{i=1}^N \pi_i B_i(O_1) \beta_1(i)$
        """)

        beta_df = pd.DataFrame(beta_trellis, columns=states, index=[f"t={t+1}: {demo_seq[t]}" for t in range(len(demo_seq))])
        st.dataframe(beta_df.style.format("{:.5e}"), use_container_width=True)

    # ==========================================
    # TAB 4: Viterbi Algorithm
    # ==========================================
    with tab_viterbi:
        render_section_title("Problem 2: Decoding Problem (Viterbi Algorithm)")
        st.markdown(r"""
        Finds the single most likely hidden state sequence $Q^* = (q_1^*, \dots, q_T^*)$ that generated the observations:
        $$V_t(j) = \max_{1 \le i \le N} \left[ V_{t-1}(i) A_{ij} \right] \cdot B_j(O_t)$$
        Backtracking is tracked through backpointer matrix $\psi_t(j) = \arg\max_i [V_{t-1}(i) A_{ij}]$.
        """)

        best_path, best_prob, v_trellis, backpointers = hmm.viterbi_algorithm(demo_seq)

        v_c1, v_c2 = st.columns([1, 2])
        with v_c1:
            render_metric_card("Viterbi Joint P(Q*, O)", f"{best_prob:.5e}", "Optimal Path Likelihood")
            st.write("**Decoded State Sequence:**")
            for t, (obs, st_name) in enumerate(zip(demo_seq, best_path)):
                st.markdown(f"**Step {t+1}** (`{obs}`): ➔ `{st_name}`")

        with v_c2:
            st.write("**Viterbi Trellis Matrix ($V_t(i)$):**")
            v_df = pd.DataFrame(v_trellis, columns=states, index=[f"t={t+1}: {demo_seq[t]}" for t in range(len(demo_seq))])
            st.dataframe(v_df.style.format("{:.5e}"), use_container_width=True)

        # Trellis Path Trajectory Plot
        state_map = {name: idx for idx, name in enumerate(states)}
        path_numeric = [state_map[s] for s in best_path]

        fig_viterbi = go.Figure()
        fig_viterbi.add_trace(go.Scatter(
            x=[f"t={t+1}<br>{demo_seq[t]}" for t in range(len(demo_seq))],
            y=path_numeric,
            mode='lines+markers',
            name='Optimal State Trajectory',
            line=dict(color='#008DDA', width=4),
            marker=dict(size=12, color='#10B981', symbol='diamond')
        ))
        fig_viterbi.update_layout(
            title="Viterbi Most Likely Hidden Health State Path",
            xaxis_title="Time Step & Observed Symptom",
            yaxis=dict(
                title="Hidden State",
                tickmode="array",
                tickvals=list(range(len(states))),
                ticktext=states
            ),
            template="plotly_dark", paper_bgcolor="#0B192C", plot_bgcolor="#112238"
        )
        st.plotly_chart(fig_viterbi, use_container_width=True)

    # ==========================================
    # TAB 5: Interactive Sequence Inference
    # ==========================================
    with tab_interactive:
        render_section_title("Interactive HMM Sequence Decoder")
        st.write("Construct an arbitrary sequence of observed symptoms to evaluate Forward, Backward, and Viterbi algorithms dynamically:")

        user_seq = st.multiselect(
            "Select Ordered Observation Sequence (in chronological order):",
            observations,
            default=["Smooth Ride", "Engine Noise", "Warning Light", "Performance Drop"]
        )

        if len(user_seq) > 0:
            if st.button("Run HMM Inference & Decoding", type="primary"):
                p_fwd, alpha_m = hmm.forward_algorithm(user_seq)
                p_bwd, beta_m = hmm.backward_algorithm(user_seq)
                v_path, v_prob, _, _ = hmm.viterbi_algorithm(user_seq)
                posteriors = hmm.compute_state_posteriors(user_seq)

                res_c1, res_c2, res_c3 = st.columns(3)
                with res_c1:
                    render_metric_card("Observation Probability", f"{p_fwd:.4e}", "Forward Likelihood")
                with res_c2:
                    render_metric_card("Viterbi Path Probability", f"{v_prob:.4e}", "Joint Likelihood")
                with res_c3:
                    render_metric_card("Most Critical Step", v_path[-1], "Final Inferred State")

                # Posterior Heatmap
                fig_post = px.imshow(
                    posteriors.T,
                    x=[f"t={t+1}: {user_seq[t]}" for t in range(len(user_seq))],
                    y=states,
                    text_auto=".1%",
                    title="Smoothed State Posterior Probabilities: P(q_t = S_i | O_1..T, λ)",
                    color_continuous_scale="Blues",
                    template="plotly_dark"
                )
                fig_post.update_layout(paper_bgcolor="#0B192C", plot_bgcolor="#112238")
                st.plotly_chart(fig_post, use_container_width=True)
        else:
            st.info("Select at least one symptom from the multiselect to run inference.")
