# geo_mapping.jl
using Pkg
Pkg.activate(".")
using Rasters, RasterDataSources, ArchGDAL
using MeshCat, GeometryBasics, Colors, CoordinateTransformations, FileIO
using AbstractGPs, KernelFunctions, Random, LinearAlgebra, Statistics
using DelimitedFiles # For ROS 2 Data Export

# ---------------------------------------------------------
# 1. SETUP ROS 2 EXPORT BRIDGE
# ---------------------------------------------------------
csv_path = joinpath(@__DIR__, "ros2_waypoints.csv")
open(csv_path, "w") do io
    write(io, "cycle,rover_id,target_x,target_y,target_z,battery,wear_index\n")
end
println("✅ ROS 2 Bridge Initialized: Logging to $csv_path")

# ---------------------------------------------------------
# 2. FETCH PHYSICAL TERRAIN (Pune, Maharashtra)
# ---------------------------------------------------------
ENV["RASTERDATASOURCES_PATH"] = joinpath(@__DIR__, "data")
raw_bounds = ((73.7, 74.0), (18.4, 18.7))
tile_paths = getraster(SRTM; bounds=raw_bounds)
lon_range = X(73.7 .. 74.0); lat_range = Y(18.4 .. 18.7)

real_dem = Raster(tile_paths[1])[lon_range, lat_range]
elevation_matrix = Array(replace_missing(real_dem, 0.0))
clean_terrain = elevation_matrix[1:4:end, 1:4:end]
new_x, new_y = size(clean_terrain)
min_alt = minimum(clean_terrain); max_alt = maximum(clean_terrain)

true_moisture = [1.0 - ((clean_terrain[i,j] - min_alt)/(max_alt - min_alt)) + 0.1*randn() for i in 1:new_x, j in 1:new_y]

# ---------------------------------------------------------
# 3. INITIALIZE HARDWARE-AWARE SWARM
# ---------------------------------------------------------
Random.seed!(42)
num_rovers = 5
X_obs = Vector{Vector{Float64}}()
y_obs = Float64[]

x_coords = range(0, 20.0, length=new_x)
y_coords = range(0, 20.0, length=new_y)
grid_points = [[x_coords[i], y_coords[j]] for j in 1:new_y for i in 1:new_x]

# 🔋 BMS & KINEMATIC STATE TRACKING
rover_positions = Vector{Vector{Float64}}()
rover_batteries = fill(100.0, num_rovers)
rover_states = fill("EXPLORING", num_rovers) # States: EXPLORING, RETURNING, CHARGING
rover_velocities = fill(0.0, num_rovers)

# Physics Limits
max_speed = 3.0
acceleration = 1.0 # Gains 1 m/s per cycle
commercial_airspace_ceiling = 620.0 # Strict No-Fly Zone above 620m absolute altitude

# The Base Station (Charging Pad)
base_station_coord = [10.0, 10.0]
base_station_z = clean_terrain[searchsortedfirst(x_coords, 10.0), searchsortedfirst(y_coords, 10.0)]

for _ in 1:num_rovers
    rx = rand(1:new_x); ry = rand(1:new_y)
    start_pos = [x_coords[rx], y_coords[ry]]
    push!(rover_positions, start_pos)
    push!(X_obs, start_pos)
    push!(y_obs, true_moisture[rx, ry])
end

kernel = 2.0 * Matern52Kernel() ∘ ScaleTransform(0.5)
f = GP(kernel)

# ---------------------------------------------------------
# 4. MESHCAT VISUALIZER WITH STL IMPORT
# ---------------------------------------------------------
vis = Visualizer()
println("\n🚀 CLICK THIS LINK TO OPEN THE 3D ENVIRONMENT:")
println("   ", vis, "\n")
sleep(8) # Give user time to open browser

setprop!(vis["/Background"], "top_color", RGB(0.05, 0.05, 0.08))
setprop!(vis["/Background"], "bottom_color", RGB(0.05, 0.05, 0.08))
setprop!(vis["/Grid"], "visible", false); setprop!(vis["/Axes"], "visible", false)

function get_visual_z(real_z)
    return (real_z - min_alt) / (max_alt - min_alt) * 4.0
end

points = Point3f[]; colors = RGBA{Float32}[]
for i in 1:new_x, j in 1:new_y
    push!(points, Point3f(x_coords[i], y_coords[j], get_visual_z(clean_terrain[i, j])))
    nz = Float32((clean_terrain[i, j] - min_alt) / (max_alt - min_alt))
    r = nz > 0.6 ? nz : 0.2f0 + nz; g = nz > 0.6 ? nz : 0.6f0 - (nz * 0.5f0); b = nz > 0.6 ? nz : 0.1f0
    push!(colors, RGBA{Float32}(r, g, b, 1.0f0))
end
setobject!(vis["environment"]["ghats"], PointCloud(points, colors), PointsMaterial(size=0.15))

# Attempt to load STL, fallback to a standard geometry if missing
# Drone Quadcopter STL from https://www.thingiverse.com/thing:1312645/files
drone_mesh = try
    load(joinpath(@__DIR__, "new_dronev_whole.stl"))
catch
    println("⚠️  No 'drone.stl' found in folder. Using default geometry.")
    Rect3f(Vec3f(-0.3, -0.3, -0.3), Vec3f(0.6, 0.6, 0.6))
end
drone_material = MeshLambertMaterial(color=colorant"red")

# ---------------------------------------------------------
# 5. THE ACTIVE LEARNING & KINEMATIC LOOP
# ---------------------------------------------------------
println("\n🚀 Initiating Hardware-Aware Swarm Logic...")

# Initializing Dynamic Hazard Map (Starts completely clear)
hazard_map = zeros(new_x, new_y)
flood_spawned = false

println("\n🚀 Initiating BMS and Airspace-Aware Swarm Logic...")

for step in 1:40
    p_fx = posterior(f(X_obs, 0.05), y_obs)
    uncertainty = reshape(var.(marginals(p_fx(grid_points))), new_x, new_y)
    
    # Cost Function: Uncertainty vs Terrain vs Hazards vs AIRSPACE
    utility = zeros(new_x, new_y)
    Threads.@threads for i in 1:new_x
        for j in 1:new_y
            actual_alt = clean_terrain[i,j]
            # Airspace Limitation: If it's too high, apply massive penalty
            airspace_penalty = actual_alt > commercial_airspace_ceiling ? 9999.0 : 0.0
            
            normalized_elevation = (actual_alt - min_alt) / (max_alt - min_alt)
            utility[i, j] = uncertainty[i, j] - (0.8 * normalized_elevation) - hazard_map[i, j] - airspace_penalty
        end
    end
    
    _, max_idx = findmax(utility)
    ideal_target = [x_coords[max_idx[1]], y_coords[max_idx[2]]]
    
    # 🧠 BMS STATE MACHINE FOR ALL ROVERS
    for active_rover in 1:num_rovers
        
        # State: CHARGING
        if rover_states[active_rover] == "CHARGING"
            rover_batteries[active_rover] += 20.0 # Charge rapidly
            if rover_batteries[active_rover] >= 100.0
                rover_batteries[active_rover] = 100.0
                rover_states[active_rover] = "EXPLORING"
                println("   🔋 Rover $active_rover is fully charged. Resuming exploration.")
            end
            continue # Skip movement while charging
        end
        
        # State: CHECK BATTERY -> RETURNING
        if rover_batteries[active_rover] < 25.0 && rover_states[active_rover] != "RETURNING"
            rover_states[active_rover] = "RETURNING"
            println("   ⚠️ Rover $active_rover reached Bingo Fuel. Returning to Base.")
        end
        
        # Determine target based on State
        current_target = rover_states[active_rover] == "RETURNING" ? base_station_coord : ideal_target
        dist_to_target = norm(rover_positions[active_rover] - current_target)
        
        # State: ARRIVED AT BASE
        if rover_states[active_rover] == "RETURNING" && dist_to_target < 0.5
            rover_states[active_rover] = "CHARGING"
            rover_velocities[active_rover] = 0.0
            continue
        end
        
        # KINEMATICS: Accelerate towards target
        if dist_to_target > 0.1
            rover_velocities[active_rover] = min(rover_velocities[active_rover] + acceleration, max_speed)
        else
            rover_velocities[active_rover] = 0.0 # Halt to take reading
        end
        
        actual_distance_moved = min(dist_to_target, rover_velocities[active_rover])
        
        if actual_distance_moved > 0
            direction_vector = normalize(current_target - rover_positions[active_rover])
            new_position = rover_positions[active_rover] + (direction_vector * actual_distance_moved)
            
            # Snap to grid
            new_x_idx = clamp(searchsortedfirst(x_coords, new_position[1]), 1, new_x)
            new_y_idx = clamp(searchsortedfirst(y_coords, new_position[2]), 1, new_y)
            actual_z = clean_terrain[new_x_idx, new_y_idx]
            
            # Drain Battery
            terrain_steepness = (actual_z - min_alt) / (max_alt - min_alt)
            rover_batteries[active_rover] -= actual_distance_moved * (1.5 + (terrain_steepness * 3.0))
            rover_positions[active_rover] = [x_coords[new_x_idx], y_coords[new_y_idx]]
            
            # If Exploring, log the data
            if rover_states[active_rover] == "EXPLORING" && rover_velocities[active_rover] == 0.0
                push!(X_obs, rover_positions[active_rover])
                push!(y_obs, true_moisture[new_x_idx, new_y_idx])
            end
            
            # Render
            rover_node = vis["rovers"]["rover_$active_rover"]

            visual_z = get_visual_z(actual_z) + 2.0
            # Add 2.0 to visual Z so the drones hover slightly above the ground
            settransform!(rover_node, compose(
    Translation(x_coords[new_x_idx], y_coords[new_y_idx], visual_z), 
    LinearMap(UniformScaling(0.005)) # Shrinks the Godzilla drone down to rover size!
))
        end
    end
    
    sleep(0.5)
end

println("\n✅ Multi-Agent Simulation Complete.")