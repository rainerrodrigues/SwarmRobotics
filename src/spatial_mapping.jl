# src/spatial_mapping.jl

function run_active_mapping()
    println("1. Simulating initial sparse telemetry...")
    Random.seed!(42)
    
    println("1. Simulating initial sparse telemetry...")
Random.seed!(42)
num_rovers = 15
X_obs = [rand(2) .* 10 for _ in 1:num_rovers] 
y_obs = [sin(x[1]*0.5) + cos(x[2]*0.5) + 0.1*randn() for x in X_obs]

spatial_kernel = 2.0 * Matern52Kernel() ∘ ScaleTransform(0.5)
f = GP(spatial_kernel)
p_fx = posterior(f(X_obs, 0.05), y_obs)

x_range = range(0, 10, length=50)
y_range = range(0, 10, length=50)
grid_points = [[x, y] for y in y_range for x in x_range]

predictions = marginals(p_fx(grid_points))
mean_moisture = reshape(mean.(predictions), 50, 50)
uncertainty   = reshape(var.(predictions), 50, 50)

println("Starting MeshCat Server inside WSL...")
vis = Visualizer()

println("\n========================================================")
println("🚀 OPEN THIS LINK IN YOUR WINDOWS WEB BROWSER:")
println("   ", vis)
println("========================================================\n")
println("Waiting 10 seconds for you to open the browser link...")
sleep(10)

# Helper function to draw/redraw the GP map as a PointCloud
function render_terrain(vis, mean_moisture, x_range, y_range)
    points = Point3f[]
    colors = RGBA{Float32}[]
    min_m, max_m = minimum(mean_moisture), maximum(mean_moisture)
    
    for i in 1:50
        for j in 1:50
            z = mean_moisture[i, j]
            push!(points, Point3f(x_range[i], y_range[j], z))
            nz = Float32((z - min_m) / (max_m - min_m))
            # Use a striking green-to-brown colormap for the "farm"
            push!(colors, RGBA{Float32}(1.0f0 - nz, nz * 0.8f0, 0.2f0, 0.6f0))
        end
    end
    
    terrain = PointCloud(points, colors)
    # Give points physical size so they render in WSL
    setobject!(vis["farm_terrain"], terrain, PointsMaterial(size=0.3))
end

println("Initializing 3D geometry...")
# Initial render of the farm map
render_terrain(vis, mean_moisture, x_range, y_range)

# Setup Rovers (Using a highly visible cyan box like your reference code)
rover_shape = Rect3f(Vec3f(-0.2, -0.2, -0.2), Vec3f(0.4, 0.4, 0.4))
rover_material = MeshLambertMaterial(color=colorant"cyan")

for (i, coord) in enumerate(X_obs)
    rover_node = vis["rovers"]["rover_$i"]
    setobject!(rover_node, rover_shape, rover_material)
    settransform!(rover_node, Translation(coord[1], coord[2], y_obs[i]))
end

println("\n🚀 Initiating Swarm Active Learning...")

# Active Learning Loop
for step in 1:10
    println("--- Cycle $step ---")
    
    # A. Target Acquisition (Hunt the mathematical variance)
    max_var_val, max_idx = findmax(uncertainty)
    target_x = x_range[max_idx[1]]
    target_y = y_range[max_idx[2]]
    
    println("   Target: Highest uncertainty at X: $(round(target_x, digits=2)), Y: $(round(target_y, digits=2))")
    
    # B. Move the active rover (Rover 1) to the new unknown coordinate
    active_rover = 1
    new_coord = [target_x, target_y]
    new_reading = sin(new_coord[1]*0.5) + cos(new_coord[2]*0.5) + 0.1*randn()

    global X_obs = push!(X_obs, new_coord)
    global y_obs = push!(y_obs, new_reading)
    
    #X_obs[active_rover] = new_coord
    #y_obs[active_rover] = new_reading
    
    # C. Update the Gaussian Process
    global p_fx = posterior(f(X_obs, 0.05), y_obs)
    global predictions = marginals(p_fx(grid_points))
    global mean_moisture = reshape(mean.(predictions), 50, 50)
    global uncertainty   = reshape(var.(predictions), 50, 50)
    
    # D. Animate the Environment
    # Redraw the terrain to show the GP learning
    render_terrain(vis, mean_moisture, x_range, y_range)
    
    # Physically move the cyan box to the new spot
    rover_node = vis["rovers"]["rover_$active_rover"]
    settransform!(rover_node, Translation(target_x, target_y, new_reading))
    
    # Pause so we can watch it happen in real-time
    sleep(1.5) 
end

println("\n✅ Active mapping complete.")
println("Press Enter in this terminal to close the server and exit...")
readline()
    
    println("\n✅ Active mapping complete.")
    println("Press Enter in this terminal to close the server and exit...")
    readline()
end
