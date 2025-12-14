"""
Quantum Optimal Transport for Battery Energy Transfer
Simplified Problem: Optimal redistribution of energy among batteries

Innovation: Treats battery charge states as quantum probability distributions
Hardware-ready: Small enough for IQM Resonance (4-8 qubits)
"""

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import Tuple, Dict
from dataclasses import dataclass

@dataclass
class Battery:
    """Individual battery state"""
    id: int
    charge_level: float  # Current charge (MWh)
    capacity: float  # Max capacity (MWh)
    efficiency: float  # Transfer efficiency (0-1)
    
    def charge_fraction(self) -> float:
        """Normalized charge level"""
        return self.charge_level / self.capacity
    
    def available_to_give(self) -> float:
        """Energy that can be transferred out"""
        return self.charge_level * 0.8  # Keep 20% reserve
    
    def capacity_to_receive(self) -> float:
        """Energy that can be received"""
        return (self.capacity - self.charge_level) * 0.9  # 90% max charge


class QuantumBatteryTransfer:
    """
    Quantum Optimal Transport for battery energy redistribution.
    
    Key Analogy (from QOT paper):
    - Source distribution μ_A: Batteries with excess charge
    - Target distribution μ_B: Batteries needing charge
    - Transport plan ρ_AB: Optimal energy flows
    - Cost C: Energy loss during transfer
    """
    
    def __init__(self, batteries: list[Battery]):
        self.batteries = batteries
        self.n_batteries = len(batteries)
        
        # Classify batteries
        avg_charge = np.mean([b.charge_fraction() for b in batteries])
        self.donors = [b for b in batteries if b.charge_fraction() > avg_charge]
        self.receivers = [b for b in batteries if b.charge_fraction() < avg_charge]
        
        print("🔋 Quantum Battery Transfer System")
        print(f"   Total batteries: {self.n_batteries}")
        print(f"   Donors: {len(self.donors)} (excess charge)")
        print(f"   Receivers: {len(self.receivers)} (need charge)")
        print(f"   Average charge: {avg_charge*100:.1f}%")
    
    def solve_classical_ot(self) -> Tuple[np.ndarray, float]:
        """
        Solve classical optimal transport problem.
        
        Formulation:
            min  Σᵢⱼ C_ij × X_ij
            s.t. Σⱼ X_ij = μ_A_i  (all excess charge is distributed)
                 Σᵢ X_ij = μ_B_j  (all charge needs are met)
                 X_ij ≥ 0
        
        Returns:
            Transport plan X and total cost
        """
        print("\n" + "="*70)
        print("METHOD 1: Classical Optimal Transport")
        print("="*70)
        
        n_donors = len(self.donors)
        n_receivers = len(self.receivers)
        
        # Source distribution: available energy from donors
        mu_source = np.array([b.available_to_give() for b in self.donors])
        mu_source = mu_source / np.sum(mu_source)  # Normalize
        
        # Target distribution: needed energy for receivers
        mu_target = np.array([b.capacity_to_receive() for b in self.receivers])
        mu_target = mu_target / np.sum(mu_target)  # Normalize
        
        print(f"   Source (donors): {mu_source}")
        print(f"   Target (receivers): {mu_target}")
        
        # Cost matrix: transfer loss between batteries
        C = self._build_cost_matrix()
        
        print(f"   Cost matrix shape: {C.shape}")
        print(f"   Average transfer cost: {np.mean(C):.4f}")
        
        # Solve optimal transport
        X = cp.Variable((n_donors, n_receivers), nonneg=True)
        
        objective = cp.Minimize(cp.sum(cp.multiply(C, X)))
        
        constraints = [
            cp.sum(X, axis=1) == mu_source,  # All donor energy distributed
            cp.sum(X, axis=0) == mu_target,  # All receiver needs met
        ]
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status != cp.OPTIMAL:
            print(f"   ⚠️ Solver status: {problem.status}")
            return None, None
        
        total_cost = problem.value
        X_optimal = X.value
        
        print(f"✅ Classical OT solution found!")
        print(f"   Optimal transport cost: {total_cost:.6f}")
        print(f"   Total energy redistributed: {np.sum(X_optimal):.4f}")
        
        # Display transfer plan
        self._display_transfer_plan(X_optimal, "Classical OT")
        
        return X_optimal, total_cost
    
    def solve_quantum_ot(self, regularization: float = 0.01) -> Tuple[np.ndarray, float]:
        """
        Solve Quantum Optimal Transport (QOT) problem.
        
        Formulation (from paper Section 1.3):
            min  Tr(C · ρ_AB)
            s.t. Tr_A(ρ_AB) = ρ_B  (marginal constraint)
                 Tr_B(ρ_AB) = ρ_A  (marginal constraint)
                 ρ_AB ≽ 0  (positive semidefinite)
                 Tr(ρ_AB) = 1  (normalized)
        
        Innovation: Density matrix ρ_AB encodes quantum correlations
        between battery states, like electron cloud interactions.
        
        Returns:
            Quantum transport plan ρ_AB and total cost
        """
        print("\n" + "="*70)
        print("METHOD 2: Quantum Optimal Transport (QOT)")
        print("="*70)
        print("⚛️  Formulating as quantum density matrix problem...")
        
        n_donors = len(self.donors)
        n_receivers = len(self.receivers)
        n_total = n_donors * n_receivers
        
        print(f"   Quantum state space: {n_donors} × {n_receivers} = {n_total}")
        print(f"   Required qubits: {int(np.ceil(np.log2(n_total)))} qubits")
        
        # Marginal distributions (quantum states)
        rho_A = np.diag([b.available_to_give() for b in self.donors])
        rho_A = rho_A / np.trace(rho_A)  # Normalize
        
        rho_B = np.diag([b.capacity_to_receive() for b in self.receivers])
        rho_B = rho_B / np.trace(rho_B)  # Normalize
        
        print(f"   ρ_A (donor states): {np.diag(rho_A)}")
        print(f"   ρ_B (receiver states): {np.diag(rho_B)}")
        
        # Cost operator (Hermitian matrix)
        C_classical = self._build_cost_matrix()
        C = np.kron(np.eye(n_donors), np.eye(n_receivers)) * np.mean(C_classical)
        
        # Add structure from classical cost
        for i in range(n_donors):
            for j in range(n_receivers):
                idx = i * n_receivers + j
                C[idx, idx] = C_classical[i, j]
        
        # Decision variable: quantum density matrix
        rho_AB = cp.Variable((n_total, n_total), hermitian=True)
        
        # Objective: minimize expected transfer cost
        objective = cp.Minimize(cp.real(cp.trace(C @ rho_AB)))
        
        # Quantum constraints
        constraints = [
            rho_AB >> 0,  # Positive semidefinite
            cp.trace(rho_AB) == 1,  # Normalized
        ]
        
        # Partial trace constraints (quantum marginals)
        # Tr_A(ρ_AB) = ρ_B: trace out donor indices
        for i in range(n_receivers):
            for j in range(n_receivers):
                partial_sum = sum([rho_AB[k*n_receivers + i, k*n_receivers + j] 
                                   for k in range(n_donors)])
                constraints.append(partial_sum == rho_B[i, j])
        
        # Tr_B(ρ_AB) = ρ_A: trace out receiver indices
        for i in range(n_donors):
            for j in range(n_donors):
                partial_sum = sum([rho_AB[i*n_receivers + k, j*n_receivers + k] 
                                   for k in range(n_receivers)])
                constraints.append(partial_sum == rho_A[i, j])
        
        # Solve quantum SDP
        problem = cp.Problem(objective, constraints)
        
        print("   Solving quantum SDP (this may take a moment)...")
        
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=5000, eps=1e-4)
            
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                total_cost = problem.value
                rho_optimal = rho_AB.value
                
                print(f"✅ Quantum OT solution found!")
                print(f"   Optimal cost: {total_cost:.6f}")
                
                # Analyze quantum properties
                self._analyze_quantum_solution(rho_optimal)
                
                # Extract classical plan from quantum solution
                X_classical = self._extract_classical_plan(rho_optimal, n_donors, n_receivers)
                self._display_transfer_plan(X_classical, "Quantum OT")
                
                return rho_optimal, total_cost
            else:
                print(f"⚠️ Solver status: {problem.status}")
                return None, None
                
        except Exception as e:
            print(f"❌ Solver error: {e}")
            return None, None
    
    def solve_quantum_sinkhorn(self, n_iterations: int = 50, lambda_reg: float = 1.0) -> Tuple[np.ndarray, float]:
        """
        Solve using Quantum Sinkhorn Algorithm (from paper Section 3.2.1).
        
        Algorithm:
            1. Initialize α_A, α_B = 0
            2. Iterate:
                ρ = exp(-I - λ(C - α_A ⊗ I - I ⊗ α_B))
                Update α_A to match marginal ρ_A
                Update α_B to match marginal ρ_B
            3. Return converged ρ
        
        This is faster than full SDP for large problems.
        """
        print("\n" + "="*70)
        print("METHOD 3: Quantum Sinkhorn Algorithm")
        print("="*70)
        print("🔥 Iterative scaling algorithm (regularized QOT)...")
        
        n_donors = len(self.donors)
        n_receivers = len(self.receivers)
        n_total = n_donors * n_receivers
        
        # Initialize
        alpha_A = np.zeros((n_donors, n_donors))
        alpha_B = np.zeros((n_receivers, n_receivers))
        
        # Cost matrix
        C_classical = self._build_cost_matrix()
        C = np.zeros((n_total, n_total))
        for i in range(n_donors):
            for j in range(n_receivers):
                idx = i * n_receivers + j
                C[idx, idx] = C_classical[i, j]
        
        # Target marginals
        rho_A = np.diag([b.available_to_give() for b in self.donors])
        rho_A = rho_A / np.trace(rho_A)
        
        rho_B = np.diag([b.capacity_to_receive() for b in self.receivers])
        rho_B = rho_B / np.trace(rho_B)
        
        print(f"   Iterations: {n_iterations}, λ={lambda_reg}")
        
        # Sinkhorn iterations
        from scipy.linalg import expm, logm
        
        convergence_history = []
        
        for k in range(n_iterations):
            # Compute current density matrix
            M = -np.eye(n_total) - lambda_reg * (
                np.kron(alpha_A, np.eye(n_receivers)) +
                np.kron(np.eye(n_donors), alpha_B) +
                C
            )
            
            try:
                rho = expm(M)
                rho = rho / np.trace(rho)  # Normalize
                
                # Compute current marginals
                rho_marg_A = np.zeros((n_donors, n_donors))
                for i in range(n_donors):
                    for j in range(n_donors):
                        rho_marg_A[i, j] = sum([rho[i*n_receivers + k, j*n_receivers + k] 
                                                for k in range(n_receivers)])
                
                rho_marg_B = np.zeros((n_receivers, n_receivers))
                for i in range(n_receivers):
                    for j in range(n_receivers):
                        rho_marg_B[i, j] = sum([rho[k*n_receivers + i, k*n_receivers + j] 
                                                for k in range(n_donors)])
                
                # Check convergence
                error_A = np.linalg.norm(rho_marg_A - rho_A, 'fro')
                error_B = np.linalg.norm(rho_marg_B - rho_B, 'fro')
                total_error = error_A + error_B
                
                convergence_history.append(total_error)
                
                if k % 10 == 0:
                    print(f"   Iteration {k:3d}: error = {total_error:.6f}")
                
                # Update alpha_A
                if error_A > 1e-6:
                    correction = rho_A @ np.linalg.pinv(rho_marg_A + 1e-10 * np.eye(n_donors))
                    alpha_A = alpha_A + (1/lambda_reg) * logm(correction)
                
                # Update alpha_B
                if error_B > 1e-6:
                    correction = rho_B @ np.linalg.pinv(rho_marg_B + 1e-10 * np.eye(n_receivers))
                    alpha_B = alpha_B + (1/lambda_reg) * logm(correction)
                
                # Check for convergence
                if total_error < 1e-5:
                    print(f"✅ Converged at iteration {k}!")
                    break
                    
            except np.linalg.LinAlgError:
                print(f"⚠️ Numerical instability at iteration {k}")
                break
        
        # Final solution
        M_final = -np.eye(n_total) - lambda_reg * (
            np.kron(alpha_A, np.eye(n_receivers)) +
            np.kron(np.eye(n_donors), alpha_B) +
            C
        )
        rho_final = expm(M_final)
        rho_final = rho_final / np.trace(rho_final)
        
        cost = np.real(np.trace(C @ rho_final))
        
        print(f"✅ Quantum Sinkhorn solution!")
        print(f"   Final cost: {cost:.6f}")
        print(f"   Final error: {convergence_history[-1]:.6f}")
        
        # Visualize convergence
        self._plot_convergence(convergence_history)
        
        # Analyze solution
        self._analyze_quantum_solution(rho_final)
        
        X_classical = self._extract_classical_plan(rho_final, n_donors, n_receivers)
        self._display_transfer_plan(X_classical, "Quantum Sinkhorn")
        
        return rho_final, cost
    
    def _build_cost_matrix(self) -> np.ndarray:
        """
        Build cost matrix for battery transfers.
        
        Cost = Energy loss during transfer
             = Base loss + Distance penalty + Efficiency loss
        """
        n_donors = len(self.donors)
        n_receivers = len(self.receivers)
        
        C = np.zeros((n_donors, n_receivers))
        
        for i, donor in enumerate(self.donors):
            for j, receiver in enumerate(self.receivers):
                # Base transfer loss (10%)
                base_loss = 0.1
                
                # Distance penalty (assume physical separation)
                distance_penalty = 0.05 * abs(donor.id - receiver.id)
                
                # Efficiency factor
                efficiency_loss = (1 - donor.efficiency) + (1 - receiver.efficiency)
                
                C[i, j] = base_loss + distance_penalty + efficiency_loss * 0.1
        
        return C
    
    def _extract_classical_plan(self, rho_AB: np.ndarray, n_donors: int, n_receivers: int) -> np.ndarray:
        """Extract classical transport plan from quantum density matrix"""
        X = np.zeros((n_donors, n_receivers))
        
        for i in range(n_donors):
            for j in range(n_receivers):
                idx = i * n_receivers + j
                X[i, j] = rho_AB[idx, idx].real  # Diagonal elements
        
        # Normalize
        X = X / np.sum(X)
        
        # Scale to match original distributions
        mu_source = np.array([b.available_to_give() for b in self.donors])
        mu_source = mu_source / np.sum(mu_source)
        X = X * np.sum(mu_source)
        
        return X
    
    def _analyze_quantum_solution(self, rho: np.ndarray):
        """Analyze quantum properties of the solution (from paper Section 2.1)"""
        print("\n⚛️  Quantum Analysis:")
        
        # Purity (Tr(ρ²))
        purity = np.trace(rho @ rho).real
        print(f"   Purity: {purity:.4f} (1.0 = pure state, <1.0 = mixed)")
        
        # Von Neumann entropy
        eigenvalues = np.linalg.eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        entropy = -np.sum(eigenvalues * np.log(eigenvalues))
        print(f"   Von Neumann entropy: {entropy:.4f} (0 = pure, high = mixed)")
        
        # Rank (number of significant eigenvalues)
        rank = np.sum(eigenvalues > 1e-6)
        print(f"   Rank: {rank} (rank-1 = extremal point, from Proposition 2.1)")
        
        # Interpretation
        if rank == 1:
            print("   💡 Rank-1 solution → Pure quantum state → Deterministic strategy")
            print("   📖 This is an extremal point (Proposition 2.1 from QOT paper)")
        elif rank <= 3:
            print("   💡 Low-rank solution → Simple strategy with few dominant modes")
        else:
            print("   💡 High-rank solution → Complex, highly entangled strategy")
    
    def _display_transfer_plan(self, X: np.ndarray, method: str):
        """Display energy transfer plan"""
        print(f"\n📊 {method} Transfer Plan:")
        print(f"   {'From (Donor)':<15} → {'To (Receiver)':<15} | Energy (MWh)")
        print(f"   {'-'*55}")
        
        for i, donor in enumerate(self.donors):
            for j, receiver in enumerate(self.receivers):
                if X[i, j] > 0.01:  # Only show significant transfers
                    energy = X[i, j] * sum([b.available_to_give() for b in self.donors])
                    print(f"   Battery {donor.id:<3} ({donor.charge_fraction()*100:5.1f}%) "
                          f"→ Battery {receiver.id:<3} ({receiver.charge_fraction()*100:5.1f}%) "
                          f"| {energy:6.3f} MWh")
    
    def _plot_convergence(self, history: list):
        """Plot Sinkhorn convergence"""
        plt.figure(figsize=(10, 6))
        plt.semilogy(history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Marginal Error (log scale)', fontsize=12)
        plt.title('Quantum Sinkhorn Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('sinkhorn_convergence.png', dpi=300)
        print("   📊 Convergence plot saved: sinkhorn_convergence.png")
    
    def visualize_transfer_network(self, X: np.ndarray, method: str):
        """Visualize battery transfer network"""
        import matplotlib.patches as mpatches
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot donors (left)
        donor_y_positions = np.linspace(0.8, 0.2, len(self.donors))
        for i, (y, donor) in enumerate(zip(donor_y_positions, self.donors)):
            charge_pct = donor.charge_fraction()
            color = plt.cm.Greens(0.3 + charge_pct * 0.7)
            circle = mpatches.Circle((0.2, y), 0.05, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(0.1, y, f"B{donor.id}\n{charge_pct*100:.0f}%", 
                   ha='right', va='center', fontsize=10)
        
        # Plot receivers (right)
        receiver_y_positions = np.linspace(0.8, 0.2, len(self.receivers))
        for j, (y, receiver) in enumerate(zip(receiver_y_positions, self.receivers)):
            charge_pct = receiver.charge_fraction()
            color = plt.cm.Reds(0.3 + (1-charge_pct) * 0.7)
            circle = mpatches.Circle((0.8, y), 0.05, color=color, ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(0.9, y, f"B{receiver.id}\n{charge_pct*100:.0f}%", 
                   ha='left', va='center', fontsize=10)
        
        # Plot transfers (arrows)
        max_transfer = np.max(X)
        for i in range(len(self.donors)):
            for j in range(len(self.receivers)):
                if X[i, j] > 0.01:
                    y1 = donor_y_positions[i]
                    y2 = receiver_y_positions[j]
                    
                    # Arrow thickness proportional to transfer amount
                    thickness = (X[i, j] / max_transfer) * 5
                    
                    ax.arrow(0.25, y1, 0.5, y2-y1, 
                            head_width=0.02, head_length=0.03,
                            fc='blue', ec='blue', alpha=0.6, linewidth=thickness)
                    
                    # Label with transfer amount
                    energy = X[i, j] * sum([b.available_to_give() for b in self.donors])
                    ax.text(0.5, (y1+y2)/2, f"{energy:.2f}", 
                           ha='center', va='bottom', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        
        ax.text(0.2, 0.95, 'DONORS\n(Excess Charge)', ha='center', fontsize=14, fontweight='bold')
        ax.text(0.8, 0.95, 'RECEIVERS\n(Need Charge)', ha='center', fontsize=14, fontweight='bold')
        ax.text(0.5, 0.05, f'{method} - Optimal Energy Transfer', 
               ha='center', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'battery_transfer_{method.replace(" ", "_").lower()}.png', dpi=300)
        print(f"   📊 Network visualization saved")
        plt.show()


# ============================================================================
# EXAMPLE: 6-Battery System
# ============================================================================

if __name__ == "__main__":
    print("⚡ QUANTUM OPTIMAL TRANSPORT FOR BATTERY ENERGY TRANSFER ⚡")
    print("="*70)
    
    # Create battery system
    batteries = [
        Battery(id=1, charge_level=14.0, capacity=16.0, efficiency=0.95),  # Nearly full
        Battery(id=2, charge_level=12.5, capacity=16.0, efficiency=0.93),  # High
        Battery(id=3, charge_level=10.0, capacity=16.0, efficiency=0.94),  # Medium-high
        Battery(id=4, charge_level=6.0, capacity=16.0, efficiency=0.92),   # Medium-low
        Battery(id=5, charge_level=3.5, capacity=16.0, efficiency=0.91),   # Low
        Battery(id=6, charge_level=1.5, capacity=16.0, efficiency=0.90),   # Nearly empty
    ]
    
    # Initialize system
    qbt = QuantumBatteryTransfer(batteries)
    
    # Method 1: Classical OT
    X_classical, cost_classical = qbt.solve_classical_ot()
    if X_classical is not None:
        qbt.visualize_transfer_network(X_classical, "Classical OT")
    
    # Method 2: Quantum OT (SDP)
    rho_quantum, cost_quantum = qbt.solve_quantum_ot()
    
    # Method 3: Quantum Sinkhorn
    rho_sinkhorn, cost_sinkhorn = qbt.solve_quantum_sinkhorn(n_iterations=30, lambda_reg=0.5)
    
    # Compare results
    print("\n" + "="*70)
    print("📊 COMPARISON OF METHODS")
    print("="*70)
    print(f"{'Method':<30} | Cost      | Status")
    print(f"{'-'*30}-|-----------|-----------")
    if cost_classical:
        print(f"{'Classical OT':<30} | {cost_classical:.6f} | Baseline")
    if cost_quantum:
        improvement = (cost_classical - cost_quantum) / cost_classical * 100
        print(f"{'Quantum OT (SDP)':<30} | {cost_quantum:.6f} | {improvement:+.2f}%")
    if cost_sinkhorn:
        improvement = (cost_classical - cost_sinkhorn) / cost_classical * 100
        print(f"{'Quantum Sinkhorn':<30} | {cost_sinkhorn:.6f} | {improvement:+.2f}%")
    
    print("\n💡 Key Insights:")
    print("   - Quantum OT captures correlations between battery states")
    print("   - Rank-1 solutions correspond to deterministic strategies (Prop. 2.1)")
    print("   - Sinkhorn provides fast approximate solutions")
    print("   - Perfect problem size for IQM Resonance (4-6 qubits)")
