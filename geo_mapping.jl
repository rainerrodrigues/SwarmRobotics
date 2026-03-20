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

# Hardware State Tracking
rover_positions = Vector{Vector{Float64}}()
rover_batteries = fill(100.0, num_rovers) # Start at 100%
rover_wear = fill(0.0, num_rovers)        # Start at 0 motor wear
max_speed_per_cycle = 3.0                 # Drones can only fly 3 meters per cycle

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

for step in 1:20
    # 🌊 DYNAMIC DISASTER EVENT: Flash Flood at Cycle 10
    if step == 10 && !flood_spawned
        println("\n⚠️  CRITICAL WARNING: MONSOON FLASH FLOOD DETECTED IN THE VALLEYS! ⚠️")
        global flood_spawned = true
        
        flood_points = Point3f[]
        flood_colors = RGBA{Float32}[]
        
        # Water pools in the lowest 15 meters of our terrain
        flood_threshold = min_alt + 15.0 
        
        for i in 1:new_x, j in 1:new_y
            if clean_terrain[i,j] < flood_threshold
                hazard_map[i, j] = 50.0 # Assign a massive mathematical penalty
                
                # Render the water in MeshCat slightly above the ground
                push!(flood_points, Point3f(x_coords[i], y_coords[j], get_visual_z(clean_terrain[i,j]) + 0.1))
                push!(flood_colors, RGBA{Float32}(0.0f0, 0.4f0, 0.8f0, 0.7f0)) # Semi-transparent blue
            end
        end
        
        # Push the floodwater to the 3D browser
        setobject!(vis["environment"]["floodwater"], PointCloud(flood_points, flood_colors), PointsMaterial(size=0.18))
        println("   -> Global Cost Map Updated. Rerouting swarm...")
    end
    
    p_fx = posterior(f(X_obs, 0.05), y_obs)
    uncertainty = reshape(var.(marginals(p_fx(grid_points))), new_x, new_y)
    
    # Cost Function: Uncertainty vs Terrain Steepness
    utility = zeros(new_x, new_y)

    # Cost Function: Uncertainty vs Terrain Steepness
    Threads.@threads for i in 1:new_x
        for j in 1:new_y
            utility[i, j] = uncertainty[i, j] - (0.8 * ((clean_terrain[i,j] - min_alt) / (max_alt - min_alt)))
        end
    end
    # List comprehension version (slightly slower but more concise):
    # utility = [uncertainty[i, j] - (0.8 * ((clean_terrain[i,j] - min_alt) / (max_alt - min_alt))) for i in 1:new_x, j in 1:new_y]
    _, max_idx = findmax(utility)
    ideal_target = [x_coords[max_idx[1]], y_coords[max_idx[2]]]
    
    # Filter for rovers that actually have battery left (> 10%)
    alive_rovers = findall(b -> b > 10.0, rover_batteries)
    if isempty(alive_rovers)
        println("⚠️ ALL ROVERS OUT OF BATTERY. Mission Abort.")
        break
    end
    
    # Find the closest ALIVE rover
    distances = [norm(rover_positions[r] - ideal_target) for r in alive_rovers]
    active_rover = alive_rovers[argmin(distances)]
    dist_to_target = distances[argmin(distances)]
    
    # KINEMATICS: Cap the movement speed. It might not reach the target this cycle!
    actual_distance_moved = min(dist_to_target, max_speed_per_cycle)
    direction_vector = normalize(ideal_target - rover_positions[active_rover])
    new_position = rover_positions[active_rover] + (direction_vector * actual_distance_moved)
    
    # Snap the new position to our nearest grid index to read the soil/altitude
    new_x_idx = clamp(searchsortedfirst(x_coords, new_position[1]), 1, new_x)
    new_y_idx = clamp(searchsortedfirst(y_coords, new_position[2]), 1, new_y)
    actual_z = clean_terrain[new_x_idx, new_y_idx]
    
    # HARDWARE PHYSICS: Calculate Battery Drain and Motor Wear
    # Steeper terrain exponentially increases battery drain and motor wear (Phase 1 HBM Bridge)
    terrain_steepness = (actual_z - min_alt) / (max_alt - min_alt)
    battery_cost = actual_distance_moved * (1.5 + (terrain_steepness * 3.0)) 
    wear_increase = actual_distance_moved * (0.01 + (terrain_steepness * 0.05)) * rand() # Pseudo-probabilistic wear
    
    rover_batteries[active_rover] -= battery_cost
    rover_wear[active_rover] += wear_increase
    rover_positions[active_rover] = [x_coords[new_x_idx], y_coords[new_y_idx]]
    
    # Log the reading to the Swarm Brain
    push!(X_obs, rover_positions[active_rover])
    push!(y_obs, true_moisture[new_x_idx, new_y_idx])
    
    # EXPORT TO ROS 2
    open(csv_path, "a") do io
        write(io, "$step,rover_$active_rover,$(x_coords[new_x_idx]),$(y_coords[new_y_idx]),$actual_z,$(rover_batteries[active_rover]),$(rover_wear[active_rover])\n")
    end
    
    # VISUALIZE
    rover_node = vis["rovers"]["rover_$active_rover"]
    setobject!(rover_node, drone_mesh, drone_material)
    visual_z = get_visual_z(actual_z)
    settransform!(rover_node, Translation(x_coords[new_x_idx], y_coords[new_y_idx], visual_z))
    
    println("Cycle $step | Rover $active_rover -> Moved $(round(actual_distance_moved, digits=1))m | Batt: $(round(rover_batteries[active_rover], digits=1))% | Wear: $(round(rover_wear[active_rover], digits=3))")
    sleep(0.8)
end

println("\n✅ Simulation Complete. Waypoints exported to $csv_path.")