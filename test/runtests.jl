# test/runtests.jl
using SwarmRobotics
using Test
using AbstractGPs
using KernelFunctions
using Statistics

@testset "SwarmRobotics.jl Spatial Mapping Tests" begin
    @testset "Variance Reduction (Active Learning)" begin
        # 1. Setup a mini GP with 2 random points
        X_obs = [[1.0, 1.0], [9.0, 9.0]]
        y_obs = [0.5, -0.5]
        
        kernel = 2.0 * Matern52Kernel() ∘ ScaleTransform(0.5)
        f = GP(kernel)
        
        # Calculate initial map variance
        p_fx_initial = posterior(f(X_obs, 0.05), y_obs)
        test_grid = [[x, y] for y in 1:10 for x in 1:10]
        initial_variance = mean(var.(marginals(p_fx_initial(test_grid))))
        
        # 2. Simulate the swarm taking a new reading in an unknown area
        push!(X_obs, [5.0, 5.0])
        push!(y_obs, 0.0)
        
        # Calculate the new map variance
        p_fx_new = posterior(f(X_obs, 0.05), y_obs)
        new_variance = mean(var.(marginals(p_fx_new(test_grid))))
        
        # 3. THE TEST: The new variance MUST be strictly less than the initial variance
        @test new_variance < initial_variance
    end
end