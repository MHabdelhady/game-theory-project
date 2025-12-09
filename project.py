import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Tuple, Dict

# ==========================================
# 1. DATA STRUCTURES: Defining the players and the market
# ==========================================

@dataclass
class ConsumerProfile:
    """Parameters defining a single large energy consumer (Follower)."""
    name: str
    marginal_value: np.ndarray    # 'a': Max Willingness to Pay ($/MWh) for the first MW
    demand_sensitivity: float     # 'b': How quickly the utility (WTP) diminishes with more consumption
    min_demand: np.ndarray        # Minimum required power (MW)
    max_capacity: np.ndarray      # Maximum physical capacity (MW)

@dataclass
class MarketSetup:
    """Parameters defining the utility company and the grid (Leader/System)."""
    T: int
    base_price_p0: np.ndarray     # Base generation cost, paid by the utility ($/MWh)
    gen_cost_c0: float            # Fixed generation cost
    gen_cost_c1: float            # Linear generation cost coefficient
    gen_cost_c2: float            # Quadratic generation cost coefficient
    grid_capacity: np.ndarray     # Maximum power the grid can handle (MW)
    capacity_penalty: float       # Cost coefficient for violating grid limit ($/MW^2)
    max_price_slope: float        # Upper limit for the leader's pricing slope (gamma)

@dataclass
class OptimizationResult:
    """The complete solution found after the Stackelberg game is solved."""
    demand_A: np.ndarray          # Hourly demand for Consumer A
    demand_B: np.ndarray          # Hourly demand for Consumer B
    total_demand: np.ndarray      # Hourly sum of all demand
    market_prices: np.ndarray     # Final hourly electricity prices
    optimal_slope: float          # Optimal pricing slope (gamma*) found by the utility
    max_social_welfare: float     # The maximized value of the social welfare function
    total_system_cost: float      # Actual total cost for the utility (Generation + Penalty)
    net_payoffs: Dict[str, float] # Net profit for each consumer (Utility - Payment)
    gross_utilities: Dict[str, float] # Total benefit/utility for each consumer (before payment)

# ==========================================
# 2. GAME SOLVER: The core logic
# ==========================================

class UtilityConsumerGame:
    """Models the Stackelberg game where the Utility sets the price slope (Leader)
    and two Consumers set their demand (Followers)."""
    
    def __init__(self, consumer_A: ConsumerProfile, consumer_B: ConsumerProfile, market: MarketSetup):
        self.A = consumer_A
        self.B = consumer_B
        self.market = market
        self.T = market.T

    def find_consumer_nash_demands(self, price_slope: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves the simultaneous Nash Equilibrium for the two consumers.
        This is done analytically for each hour 't' assuming elastic demand.
        The result is the optimal demand for A and B given the price_slope (gamma).
        """
        d_A = np.zeros(self.T)
        d_B = np.zeros(self.T)
        
        # The 2x2 matrix that defines the interaction between the two consumers' demand
        M = np.array([
            [self.A.demand_sensitivity + 2*price_slope, price_slope],
            [price_slope, self.B.demand_sensitivity + 2*price_slope]
        ])
        det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
        
        for t in range(self.T):
            # Right-hand side (RHS) of the Nash condition equation: Max_WTP - Base_Cost
            rhs = np.array([
                self.A.marginal_value[t] - self.market.base_price_p0[t],
                self.B.marginal_value[t] - self.market.base_price_p0[t]
            ])
            
            if det == 0:
                vals = np.zeros(2) # Should not happen unless parameters are poor
            else:
                vals = np.linalg.solve(M, rhs)
            
            # Clip the calculated demand to stay within the consumers' physical limits
            d_A[t] = np.clip(vals[0], self.A.min_demand[t], self.A.max_capacity[t])
            d_B[t] = np.clip(vals[1], self.B.min_demand[t], self.B.max_capacity[t])
            
        return d_A, d_B

    def calculate_social_welfare(self, price_slope: float) -> float:
        d_A, d_B = self.find_consumer_nash_demands(price_slope)
        total_demand = d_A + d_B
        
        # 1. Gross Utility (Benefit): The intrinsic value consumers get from using the power
        util_A = np.sum(self.A.marginal_value * d_A - 0.5 * self.A.demand_sensitivity * d_A**2)
        util_B = np.sum(self.B.marginal_value * d_B - 0.5 * self.B.demand_sensitivity * d_B**2)
        
        # 2. System Cost: Cost to generate power + cost for capacity violations
        gen_cost = np.sum(
            self.market.gen_cost_c0 + 
            self.market.gen_cost_c1 * total_demand + 
            0.5 * self.market.gen_cost_c2 * total_demand**2
        )
        
        # Capacity Violation Penalty
        violations = np.maximum(0, total_demand - self.market.grid_capacity)
        penalty = self.market.capacity_penalty * np.sum(violations**2)
        
        return (util_A + util_B) - (gen_cost + penalty)

    def find_optimal_pricing_slope(self) -> OptimizationResult:
        """
        The main optimization step: The utility searches for the price slope (gamma)
        that maximizes the calculated Social Welfare.
        """
        # The minimize_scalar function minimizes, so we flip the sign of the objective
        objective = lambda g: -1.0 * self.calculate_social_welfare(g)
        
        # Use a built-in 1D minimization tool to find the optimal gamma
        res = minimize_scalar(
            objective, 
            bounds=(0, self.market.max_price_slope), 
            method='bounded'
        )
        
        optimal_slope = res.x
        max_welfare = -res.fun # Flip the sign back to get the max welfare value
        
        # --- Final Equilibrium State Calculations (at the optimal slope) ---
        d_A, d_B = self.find_consumer_nash_demands(optimal_slope)
        total_demand = d_A + d_B
        
        # Final electricity price paid by consumers
        prices = self.market.base_price_p0 + optimal_slope * total_demand
        
        # Recalculate Gross Utilities and Payments for the final results summary
        util_A = np.sum(self.A.marginal_value * d_A - 0.5 * self.A.demand_sensitivity * d_A**2)
        util_B = np.sum(self.B.marginal_value * d_B - 0.5 * self.B.demand_sensitivity * d_B**2)
        
        # Net Payoff = Gross Utility - Total Payment
        net_payoff_A = util_A - np.sum(prices * d_A)
        net_payoff_B = util_B - np.sum(prices * d_B)
        
        # Calculate final System Cost
        gen_cost = np.sum(self.market.gen_cost_c0 + self.market.gen_cost_c1 * total_demand + 0.5 * self.market.gen_cost_c2 * total_demand**2)
        penalty = self.market.capacity_penalty * np.sum(np.maximum(0, total_demand - self.market.grid_capacity)**2)
        
        return OptimizationResult(
            d_A, d_B, total_demand, prices, optimal_slope, max_welfare, 
            gen_cost + penalty, 
            {'A': net_payoff_A, 'B': net_payoff_B},
            {'A': util_A, 'B': util_B}
        )
    
    def plot_optimization_landscape(self, pts=100):
        """Calculates the social welfare across a range of possible price slopes."""
        slopes = np.linspace(0, self.market.max_price_slope, pts)
        welfares = [self.calculate_social_welfare(g) for g in slopes]
        return slopes, np.array(welfares)

# ==========================================
# 3. ANALYSIS & VISUALIZATION: Reporting the outcome
# ==========================================

def print_detailed_report(results: OptimizationResult, market: MarketSetup):
    """Prints a detailed, formatted summary of all equilibrium results."""
    
    print("\n" + "="*70)
    print("UTILITY-CONSUMER EQUILIBRIUM: DETAILED REPORT")
    print("="*70)
    
    print(f"\n1. UTILITY'S OPTIMAL CONTROL (LEADER)")
    print("-" * 40)
    print(f"  Optimal Pricing Slope (γ*): {results.optimal_slope:.6f} $/MWh/MW")
    print(f"  Search Range for γ:  [0, {market.max_price_slope}]")
    
    print(f"\n2. OVERALL MARKET PERFORMANCE (OBJECTIVES)")
    print("-" * 40)
    print(f"  MAX Social Welfare: ${results.max_social_welfare:,.2f} (Maximized Objective)")
    print(f"  Total System Cost: ${results.total_system_cost:,.2f} (Generation + Penalty)")
    total_utility = results.gross_utilities['A'] + results.gross_utilities['B']
    print(f"  Total Consumer Utility: ${total_utility:,.2f}  (Gross Benefit)")
    
    print(f"\n3. CONSUMER OUTCOMES (PAYOFFS & USAGE)")
    print("-" * 40)
    # Consumer A
    print(f"  Consumer A:")
    cost_A = results.gross_utilities['A'] - results.net_payoffs['A']
    print(f"  - Net Payoff (Profit): ${results.net_payoffs['A']:,.2f}")
    print(f"  - Gross Utility (Benefit):  ${results.gross_utilities['A']:,.2f}")
    print(f"  - Total Cost Paid: ${cost_A:,.2f}")
    print(f"  - Total Energy Consumed:  {np.sum(results.demand_A):,.2f} MWh")
    
    # Consumer B
    print(f"  Consumer B:")
    cost_B = results.gross_utilities['B'] - results.net_payoffs['B']
    print(f"  - Net Payoff (Profit): ${results.net_payoffs['B']:,.2f}")
    print(f" - Gross Utility (Benefit): ${results.gross_utilities['B']:,.2f}")
    print(f"  - Total Cost Paid:  ${cost_B:,.2f}")
    print(f" - Total Energy Consumed:  {np.sum(results.demand_B):,.2f} MWh")

    print(f"\n4. HOURLY MARKET STATISTICS")
    print("-" * 40)
    print(f" Average Price:  ${np.mean(results.market_prices):.2f} /MWh")
    print(f"  Peak Price: ${np.max(results.market_prices):.2f} /MWh")
    print(f" Avg Base Cost (p0): ${np.mean(market.base_price_p0):.2f} /MWh")
    
    print(f"\n5. GRID CONGESTION STATUS")
    print("-" * 40)
    peak_load = np.max(results.total_demand)
    capacity = market.grid_capacity[0]
    print(f"  Peak System Demand: {peak_load:.2f} MW")
    print(f"  Grid Line Capacity: {capacity:.2f} MW")
    if peak_load > capacity:
        print(f"  STATUS: CAPACITY VIOLATION ({peak_load - capacity:.2f} MW over)")
    else:
        print(f"  STATUS: SAFE ({(capacity - peak_load)/capacity*100:.1f}% margin)")
    print("="*70 + "\n")


def generate_visual_report(game: UtilityConsumerGame, results: OptimizationResult):
    """Generates a series of plots to visualize the game outcomes."""
    hours = np.arange(game.T)
    
    # --- FIGURE 1: HOURLY DEMAND AND GRID LIMITS ---
    plt.figure(figsize=(10, 6))
    plt.bar(hours, results.demand_A, label=game.A.name, color='#1f77b4', alpha=0.8, width=0.6)
    plt.bar(hours, results.demand_B, bottom=results.demand_A, label=game.B.name, color='#ff7f0e', alpha=0.8, width=0.6)
    plt.plot(hours, results.total_demand, 'k--', linewidth=1.5, label='Total Demand Trend')
    plt.axhline(game.market.grid_capacity[0], color='r', linestyle=':', linewidth=2, label='Grid Capacity Limit')
    plt.title("Hourly Energy Consumption and Grid Load", fontsize=14)
    plt.xlabel("Hour of Day")
    plt.ylabel("Demand (MW)")
    plt.xticks(hours)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show() # 

    # --- FIGURE 2: MARKET PRICE VS. COST ---
    plt.figure(figsize=(10, 6))
    plt.plot(hours, game.market.base_price_p0, 'k--', label='Utility Base Generation Cost ($p_0$)', alpha=0.6)
    plt.plot(hours, results.market_prices, 'g-o', linewidth=2, label='Final Market Price ($P_t$)', markersize=4)
    plt.plot(hours, game.A.marginal_value, color='#1f77b4', linestyle=':', label=f'{game.A.name} Max Willingness to Pay')
    plt.plot(hours, game.B.marginal_value, color='#ff7f0e', linestyle=':', label=f'{game.B.name} Max Willingness to Pay')
    plt.title("Electricity Price Dynamics", fontsize=14)
    plt.xlabel("Hour of Day")
    plt.ylabel("Price ($/MWh)")
    plt.xticks(hours)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- FIGURE 3: UTILITY'S OPTIMIZATION LANDSCAPE ---
    plt.figure(figsize=(10, 6))
    slopes, welfares = game.plot_optimization_landscape()
    plt.plot(slopes, welfares, 'b-', linewidth=2, label='Social Welfare')
    plt.axvline(results.optimal_slope, color='r', linestyle='--', label=f'Optimal Slope (γ*) = {results.optimal_slope:.2f}')
    plt.plot([results.optimal_slope], [results.max_social_welfare], 'ro', markersize=8, label='Maximum Welfare Point')
    plt.title("Utility's Optimization: Social Welfare vs. Pricing Slope (Gamma)", fontsize=14)
    plt.xlabel("Price Slope ($\gamma$ in \$/MWh/MW)")
    plt.ylabel("Total Social Welfare ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- FIGURE 4: TOTAL CONSUMPTION SUMMARY ---
    plt.figure(figsize=(8, 6))
    total_A = np.sum(results.demand_A)
    total_B = np.sum(results.demand_B)
    bars = plt.bar([game.A.name, game.B.name], [total_A, total_B], color=['#1f77b4', '#ff7f0e'], width=0.5)
    plt.bar_label(bars, fmt='%.0f MWh', padding=3)
    plt.title("Total Daily Energy Consumption Summary", fontsize=14)
    plt.ylabel("Total Energy (MWh)")
    plt.ylim(0, max(total_A, total_B) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==========================================
# 4. RUN EXPERIMENT: Example Simulation
# ==========================================

if __name__ == "__main__":
    
    # --- Define Simulation Parameters ---
    T = 24
    hours = np.arange(T)
    
    # Base Price Curve: The cost for the utility to generate power (Low at night, high during peak day)
    # p0: ~$30 night, ~$45 peak (around hour 14)
    base_price_curve = 30 + 15 * np.exp(-((hours - 14)**2) / 50) 
    
    # Consumer A (e.g., Data Center): High WTP, slightly flexible
    # WTP is high but dips slightly during peak cost hours
    wtp_A = 85 - 5 * np.exp(-((hours - 14)**2) / 50)
    
    consumer_A = ConsumerProfile(
        name="Data Center", marginal_value=wtp_A, demand_sensitivity=0.6, 
        min_demand=np.zeros(T), max_capacity=np.ones(T) * 100
    )
    
    # Consumer B (e.g., Steel Plant): Constant, but slightly less aggressive WTP
    # Constant WTP across the day due to continuous operation
    consumer_B = ConsumerProfile(
        name="Steel Plant", marginal_value=82 * np.ones(T), demand_sensitivity=0.7, 
        min_demand=np.zeros(T), max_capacity=np.ones(T) * 80
    )
    
    market_setup = MarketSetup(
        T=T, base_price_p0=base_price_curve, 
        gen_cost_c0=50, gen_cost_c1=28, gen_cost_c2=0.08, 
        grid_capacity=np.ones(T) * 120, capacity_penalty=200, max_price_slope=10.0
    )
    
    # --- Execute Game ---
    game_model = UtilityConsumerGame(consumer_A, consumer_B, market_setup)
    results = game_model.find_optimal_pricing_slope()
    
    # --- Generate Report and Visuals ---
    print_detailed_report(results, market_setup)
    generate_visual_report(game_model, results)
