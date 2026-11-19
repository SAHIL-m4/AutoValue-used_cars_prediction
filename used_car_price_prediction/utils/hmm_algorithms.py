"""
Mathematical and algorithmic implementation of Markov and Hidden Markov Models (HMM).
Implements the Forward Algorithm, Backward Algorithm, Viterbi Algorithm, and Markov Transition Solvers.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

class MarkovModel:
    """Discrete First-Order Markov Chain Model."""

    def __init__(self, states: List[str], transition_matrix: np.ndarray, initial_probs: Optional[np.ndarray] = None):
        self.states = states
        self.N = len(states)
        self.A = np.array(transition_matrix, dtype=float)

        # Normalize rows to ensure valid stochastic matrix
        row_sums = self.A.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.A = self.A / row_sums

        if initial_probs is None:
            self.pi = np.ones(self.N) / self.N
        else:
            self.pi = np.array(initial_probs, dtype=float)
            self.pi = self.pi / self.pi.sum()

    def get_k_step_transition(self, k: int) -> np.ndarray:
        """Calculates A^k transition matrix."""
        return np.linalg.matrix_power(self.A, k)

    def compute_steady_state(self) -> np.ndarray:
        """
        Solves for stationary distribution pi_ss such that:
        pi_ss * A = pi_ss  and sum(pi_ss) = 1.
        Uses eigenvector decomposition of A.T corresponding to eigenvalue 1.
        """
        eigvals, eigvecs = np.linalg.eig(self.A.T)
        # Find index of eigenvalue closest to 1.0
        idx = np.argmin(np.abs(eigvals - 1.0))
        stationary = np.real(eigvecs[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)

    def simulate_sequence(self, steps: int = 10, seed: Optional[int] = None) -> List[str]:
        """Simulates a state transition path over given number of steps."""
        if seed is not None:
            np.random.seed(seed)
        sequence = []
        curr_state = np.random.choice(self.N, p=self.pi)
        sequence.append(self.states[curr_state])

        for _ in range(steps - 1):
            curr_state = np.random.choice(self.N, p=self.A[curr_state])
            sequence.append(self.states[curr_state])
        return sequence


class HiddenMarkovModel:
    """
    Discrete Hidden Markov Model (HMM) lambda = (A, B, pi).
    States: S = {S_1, ..., S_N}
    Observations: V = {v_1, ..., v_M}
    """

    def __init__(
        self,
        states: List[str],
        observations: List[str],
        transition_matrix: np.ndarray,
        emission_matrix: np.ndarray,
        initial_probs: Optional[np.ndarray] = None
    ):
        self.states = states
        self.observations = observations
        self.N = len(states)
        self.M = len(observations)

        # Transition matrix A: N x N
        self.A = np.array(transition_matrix, dtype=float)
        row_sums_A = self.A.sum(axis=1, keepdims=True)
        row_sums_A[row_sums_A == 0] = 1.0
        self.A = self.A / row_sums_A

        # Emission matrix B: N x M
        self.B = np.array(emission_matrix, dtype=float)
        row_sums_B = self.B.sum(axis=1, keepdims=True)
        row_sums_B[row_sums_B == 0] = 1.0
        self.B = self.B / row_sums_B

        # Initial distribution pi
        if initial_probs is None:
            self.pi = np.ones(self.N) / self.N
        else:
            self.pi = np.array(initial_probs, dtype=float)
            self.pi = self.pi / self.pi.sum()

        self.obs_to_idx = {obs: idx for idx, obs in enumerate(self.observations)}
        self.state_to_idx = {st: idx for idx, st in enumerate(self.states)}

    def forward_algorithm(self, obs_sequence: List[str]) -> Tuple[float, np.ndarray]:
        """
        Executes the Forward Algorithm to compute P(O | lambda).
        alpha_t(i) = P(O_1, ..., O_t, q_t = S_i | lambda)
        Returns:
            total_prob: Total observation sequence probability P(O | lambda)
            alpha_trellis: (T x N) matrix of alpha values
        """
        T = len(obs_sequence)
        if T == 0:
            return 0.0, np.zeros((0, self.N))

        obs_indices = [self.obs_to_idx.get(o, 0) for o in obs_sequence]
        alpha = np.zeros((T, self.N))

        # Initialization at t = 0
        o_0 = obs_indices[0]
        for i in range(self.N):
            alpha[0, i] = self.pi[i] * self.B[i, o_0]

        # Induction for t = 1 ... T-1
        for t in range(1, T):
            o_t = obs_indices[t]
            for j in range(self.N):
                sum_trans = np.sum(alpha[t - 1, :] * self.A[:, j])
                alpha[t, j] = sum_trans * self.B[j, o_t]

        # Termination
        total_prob = float(np.sum(alpha[T - 1, :]))
        return total_prob, alpha

    def backward_algorithm(self, obs_sequence: List[str]) -> Tuple[float, np.ndarray]:
        """
        Executes the Backward Algorithm to compute beta_t(i).
        beta_t(i) = P(O_{t+1}, ..., O_T | q_t = S_i, lambda)
        Returns:
            total_prob: Total observation sequence probability
            beta_trellis: (T x N) matrix of beta values
        """
        T = len(obs_sequence)
        if T == 0:
            return 0.0, np.zeros((0, self.N))

        obs_indices = [self.obs_to_idx.get(o, 0) for o in obs_sequence]
        beta = np.zeros((T, self.N))

        # Initialization at t = T - 1
        beta[T - 1, :] = 1.0

        # Induction backwards from t = T-2 down to 0
        for t in range(T - 2, -1, -1):
            o_next = obs_indices[t + 1]
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, o_next] * beta[t + 1, :])

        # Termination (at t = 0)
        o_0 = obs_indices[0]
        total_prob = float(np.sum(self.pi * self.B[:, o_0] * beta[0, :]))
        return total_prob, beta

    def viterbi_algorithm(self, obs_sequence: List[str]) -> Tuple[List[str], float, np.ndarray, np.ndarray]:
        """
        Executes the Viterbi Algorithm to find the optimal hidden state sequence Q*.
        V_t(j) = max_i [V_{t-1}(i) * A_ij] * B_j(O_t)
        Returns:
            best_path_states: Most likely sequence of hidden states
            best_path_prob: Joint probability of best path and observations
            viterbi_trellis: (T x N) matrix of Viterbi delta values
            backpointers: (T x N) matrix of psi indices
        """
        T = len(obs_sequence)
        if T == 0:
            return [], 0.0, np.zeros((0, self.N)), np.zeros((0, self.N), dtype=int)

        obs_indices = [self.obs_to_idx.get(o, 0) for o in obs_sequence]
        viterbi = np.zeros((T, self.N))
        backpointer = np.zeros((T, self.N), dtype=int)

        # Initialization
        o_0 = obs_indices[0]
        for i in range(self.N):
            viterbi[0, i] = self.pi[i] * self.B[i, o_0]
            backpointer[0, i] = 0

        # Recursion
        for t in range(1, T):
            o_t = obs_indices[t]
            for j in range(self.N):
                trans_probs = viterbi[t - 1, :] * self.A[:, j]
                max_prev_state = np.argmax(trans_probs)
                max_val = trans_probs[max_prev_state]

                viterbi[t, j] = max_val * self.B[j, o_t]
                backpointer[t, j] = max_prev_state

        # Termination
        best_last_state = np.argmax(viterbi[T - 1, :])
        best_path_prob = float(viterbi[T - 1, best_last_state])

        # Path Backtracking
        best_path_idx = [0] * T
        best_path_idx[T - 1] = best_last_state
        for t in range(T - 1, 0, -1):
            best_path_idx[t - 1] = backpointer[t, best_path_idx[t]]

        best_path_states = [self.states[idx] for idx in best_path_idx]
        return best_path_states, best_path_prob, viterbi, backpointer

    def compute_state_posteriors(self, obs_sequence: List[str]) -> np.ndarray:
        """
        Computes gamma_t(i) = P(q_t = S_i | O, lambda) using Forward and Backward trellises.
        gamma_t(i) = [alpha_t(i) * beta_t(i)] / sum_j [alpha_t(j) * beta_t(j)]
        """
        _, alpha = self.forward_algorithm(obs_sequence)
        _, beta = self.backward_algorithm(obs_sequence)
        
        gamma = alpha * beta
        normalizer = gamma.sum(axis=1, keepdims=True)
        normalizer[normalizer == 0] = 1e-12
        gamma = gamma / normalizer
        return gamma
