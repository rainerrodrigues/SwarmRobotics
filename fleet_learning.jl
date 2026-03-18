# fleet_learning.jl
using Turing
using Distributions
using DataFrames

# Defining the Hierarchical Model
@model function rover_degradation(rover_ids, temperature_spikes, num_rovers)
    # Global Priors: The "factory baseline" for all motors
    global_mean ~ Normal(5.0, 2.0)  # Average expected spikes
    global_variance ~ truncated(Cauchy(0, 1), 0, Inf) 
    
    # Local Parameters: Individual degradation for each rover
    # We use a vector of length `num_rovers` mapped to the global prior
    local_rates ~ filldist(Normal(global_mean, global_variance), num_rovers)
    
    # Observations: Tie the actual data to the local rover rates
    for i in eachindex(temperature_spikes)
        rover_idx = rover_ids[i]
        # Ensuring the rate is positive for a Poisson distribution
        rate = max(0.01, local_rates[rover_idx]) 
        temperature_spikes[i] ~ Poisson(rate)
    end
end

# Generating Simulated Fleet Telemetry
println("Generating fleet telemetry...")
# Let's say we have 3 rovers. Rover 2 is operating in harsh terrain.
telemetry_data = DataFrame(
    RoverID = [1, 1, 1, 2, 2, 2, 3, 3, 3],
    Spikes  = [4, 5, 4, 12, 14, 11, 5, 6, 4] 
)

# Instantiating and Running the Model
num_rovers = length(unique(telemetry_data.RoverID))
model = rover_degradation(telemetry_data.RoverID, telemetry_data.Spikes, num_rovers)

# Performing Inference (Active Learning)
println("Running NUTS Sampler...")
# We use the No-U-Turn Sampler (NUTS) to compute the probabilities
chain = sample(model, NUTS(), 1000)

# Displaying the results
display(chain)