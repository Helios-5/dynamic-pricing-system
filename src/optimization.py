import pulp

def optimize_price(predicted_price, current_utilization, base_retention=1.0):
    """
    Optimizes the price multiplier using Linear Programming (PuLP).
    
    Objective: Maximize Price Multiplier (proxy for Revenue in this simple model, 
               assuming inelastic demand within small range)
    
    Constraints:
    1. Customer Retention Rate >= 0.8
       Assumption: Retention drops by 0.1 for every 1.0 increase in multiplier.
       Retention = Base_Retention - 0.1 * (Price - 1) >= 0.8
    
    2. Driver Utilization <= 0.9
       Assumption: Utilization drops by 0.05 for every 1.0 increase in multiplier (due to lower demand).
       Utilization = Current_Utilization - 0.05 * (Price - 1) <= 0.9
       
    3. Price within reasonable bounds of the predicted price (e.g., +/- 20%)
    """
    
    # Create the LP problem
    prob = pulp.LpProblem("Dynamic_Pricing_Optimization", pulp.LpMaximize)
    
    # Decision Variable: Price Multiplier
    # Bounds: 1.0 to 5.0
    price = pulp.LpVariable("Price_Multiplier", lowBound=1.0, upBound=5.0, cat='Continuous')
    
    # Objective: Maximize Price (simplified revenue maximization)
    prob += price
    
    # Constraint 1: Retention >= 0.8
    # Retention = 1.0 - 0.1 * (price - 1) >= 0.8
    # => 1.0 - 0.1*price + 0.1 >= 0.8
    # => 1.1 - 0.1*price >= 0.8
    # => 0.1*price <= 0.3
    # => price <= 3.0 (Wait, this logic implies higher price = lower retention, so we want to keep price low enough)
    # Let's use the equation directly in pulp
    prob += (base_retention - 0.1 * (price - 1)) >= 0.8, "Min_Retention"
    
    # Constraint 2: Utilization <= 0.9
    # Utilization = Current_Utilization - 0.05 * (price - 1) <= 0.9
    # If Current_Utilization is already > 0.9, we need to increase price to reduce it.
    # If Current_Utilization is low, this constraint is easily met.
    prob += (current_utilization - 0.05 * (price - 1)) <= 0.9, "Max_Utilization"
    
    # Constraint 3: Trust Region (stay close to ML prediction)
    prob += price <= predicted_price * 1.2, "Upper_Trust"
    prob += price >= predicted_price * 0.8, "Lower_Trust"
    
    # Solve
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    if pulp.LpStatus[status] == 'Optimal':
        return round(pulp.value(price), 2)
    else:
        return predicted_price # Fallback

if __name__ == "__main__":
    # Test
    p1 = optimize_price(predicted_price=2.0, current_utilization=0.95)
    print(f"Pred: 2.0, Util: 0.95 -> Opt: {p1}")
    
    p2 = optimize_price(predicted_price=1.5, current_utilization=0.5)
    print(f"Pred: 1.5, Util: 0.5 -> Opt: {p2}")
