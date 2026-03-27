# geo_mapping.jl
using Pkg
Pkg.activate(".")
using Rasters, RasterDataSources, ArchGDAL
using MeshCat, GeometryBasics, Colors, CoordinateTransformations, FileIO, MeshIO
using AbstractGPs, KernelFunctions, Random, LinearAlgebra, Statistics

include(joinpath(@__DIR__, "Scenarios", "config.jl"))
ACTIVE_SCENARIO = "CYCLONE_FLOOD" # Options: "EARTHQUAKE", "CYCLONE_FLOOD"
victim_coords, wind_penalty, cv_vision_radius, flood_level = load_scenario(ACTIVE_SCENARIO)
num_victims = length(victim_coords)

# ---------------------------------------------------------
# 1. FETCH PHYSICAL TERRAIN 
# ---------------------------------------------------------
ENV["RASTERDATASOURCES_PATH"] = joinpath(@__DIR__, "data")
raw_bounds = ((73.7, 74.0), (18.4, 18.7))
tile_paths = getraster(SRTM; bounds=raw_bounds)
lon_range = X(73.7 .. 74.0); lat_range = Y(18.4 .. 18.7)

real_dem = Raster(tile_paths[1])[lon_range, lat_range]
elevation_matrix = Array(replace_missing(real_dem, 0.0))
clean_terrain = elevation_matrix[1:end, 1:end]
new_x, new_y = size(clean_terrain)
min_alt = minimum(clean_terrain); max_alt = maximum(clean_terrain)

true_moisture = [1.0 - ((clean_terrain[i,j] - min_alt)/(max_alt - min_alt)) + 0.1*randn() for i in 1:new_x, j in 1:new_y]

# ---------------------------------------------------------
# 2. INITIALIZE SWARM & DYNAMIC BASE STATIONS
# ---------------------------------------------------------
Random.seed!(42)
num_rovers = 5
X_obs = Vector{Vector{Float64}}()
y_obs = Float64[]

x_coords = range(0, 20.0, length=new_x)
y_coords = range(0, 20.0, length=new_y)
grid_points = [[x_coords[i], y_coords[j]] for j in 1:new_y for i in 1:new_x]

rover_positions = Vector{Vector{Float64}}()
rover_batteries = fill(100.0, num_rovers)
rover_states = fill("EXPLORING", num_rovers)
rover_hover_alts = fill(2.0, num_rovers) 

# 🧠 MOTION PLANNING: Upgrade to Vector Velocities [vx, vy]
rover_velocities = [ [0.0, 0.0] for _ in 1:num_rovers ]
max_speed = 3.0
dt = 0.5 # Time delta for physics integration

commercial_airspace_ceiling = 620.0 
global base_station_coord = [10.0, 10.0]
global base_station_status = (ACTIVE_SCENARIO == "EARTHQUAKE") ? "UNVERIFIED" : "SAFE"
secondary_base_coord = [2.0, 18.0] 

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
# 3. MESHCAT VISUALIZER
# ---------------------------------------------------------
vis = Visualizer()
println("\n🚀 OPEN 3D ENVIRONMENT: ", vis)
sleep(5) 

setprop!(vis["/Background"], "top_color", RGB(0.05, 0.05, 0.08))
setprop!(vis["/Background"], "bottom_color", RGB(0.05, 0.05, 0.08))

setprop!(vis["/Grid"], "visible", false)
setprop!(vis["/Axes"], "visible", false)
function get_visual_z(real_z) return (real_z - min_alt) / (max_alt - min_alt) * 4.0 end

# To center the mapping to the center of simulation environment
vis_offset = 10.0

points = Point3f[]; colors = RGBA{Float32}[]
for i in 1:new_x, j in 1:new_y
    push!(points, Point3f(x_coords[i] - vis_offset, y_coords[j] - vis_offset, get_visual_z(clean_terrain[i, j])))
    nz = Float32((clean_terrain[i, j] - min_alt) / (max_alt - min_alt))
    push!(colors, RGBA{Float32}(nz > 0.6 ? nz : 0.2f0 + nz, nz > 0.6 ? nz : 0.6f0 - (nz * 0.5f0), nz > 0.6 ? nz : 0.1f0, 1.0f0))
end
setobject!(vis["environment"]["ghats"], PointCloud(points, colors), PointsMaterial(size=0.08))

base_station_z = clean_terrain[searchsortedfirst(x_coords, base_station_coord[1]), searchsortedfirst(y_coords, base_station_coord[2])]
setobject!(vis["environment"]["base"], Cylinder(Point3f(0,0,0), Point3f(0,0,0.2), 0.8f0), MeshLambertMaterial(color=colorant"yellow"))
settransform!(vis["environment"]["base"], Translation(base_station_coord[1] - vis_offset, base_station_coord[2] - vis_offset, get_visual_z(base_station_z)))

hazard_map = zeros(new_x, new_y)
hazard_map = zeros(new_x, new_y)
if flood_level > 0.0
    flood_points = Point3f[]; flood_colors = RGBA{Float32}[]
    threshold = min_alt + flood_level
    for i in 1:new_x, j in 1:new_y
        if clean_terrain[i,j] < threshold
            hazard_map[i, j] = 50.0 
            # Apply offset to Floodwater
            push!(flood_points, Point3f(x_coords[i] - vis_offset, y_coords[j] - vis_offset, get_visual_z(clean_terrain[i,j]) + 0.1))
            push!(flood_colors, RGBA{Float32}(0.0f0, 0.4f0, 0.8f0, 0.7f0))
        end
    end
    setobject!(vis["environment"]["flood"], PointCloud(flood_points, flood_colors), PointsMaterial(size=0.09))
end

victim_statuses = fill("SEARCHING", num_victims)
assigned_rescuer = fill(0, num_victims)
for (idx, coord) in enumerate(victim_coords)
    hz = clean_terrain[searchsortedfirst(x_coords, coord[1]), searchsortedfirst(y_coords, coord[2])]
    setobject!(vis["environment"]["victim_$idx"], Sphere(Point3f(0,0,0), 0.15f0), MeshLambertMaterial(color=colorant"orange"))
    settransform!(vis["environment"]["victim_$idx"], Translation(coord[1] - vis_offset, coord[2] - vis_offset, get_visual_z(hz)))
end

drone_mesh = try load(joinpath(@__DIR__, "new_dronev_whole.stl")) catch; Rect3f(Vec3f(-0.3, -0.3, -0.3), Vec3f(0.6, 0.6, 0.6)) end
for r in 1:num_rovers setobject!(vis["rovers"]["rover_$r"], drone_mesh, MeshLambertMaterial(color=colorant"red")) end

# ---------------------------------------------------------
# 4. PARALLELIZED & VECTORIZED ENGINE
# ---------------------------------------------------------
println("\n=================================================")
println("🚀 S.A.R. FLEET: ADVANCED SWARM DYNAMICS ENGAGED")
println("=================================================")

for step in 1:150 
    println("\n--- Cycle $step ---")
    p_fx = posterior(f(X_obs, 0.05), y_obs)
    uncertainty = reshape(var.(marginals(p_fx(grid_points))), new_x, new_y)
    utility = zeros(new_x, new_y)
    
    # optimized parallization with @inbounds and @simd strip safety checks for raw C-level CPU performance
    Threads.@threads for i in 1:new_x
        @inbounds @simd for j in 1:new_y
            actual_alt = clean_terrain[i,j]
            utility[i, j] = uncertainty[i, j] - (0.8 * ((actual_alt - min_alt) / (max_alt - min_alt))) - hazard_map[i, j] - (actual_alt > commercial_airspace_ceiling ? 9999.0 : 0.0)
        end
    end
    
    _, max_idx = findmax(utility)
    global_exploration_target = [x_coords[max_idx[1]], y_coords[max_idx[2]]]
    
    for active_rover in 1:num_rovers
        current_target = global_exploration_target
        active_victim_idx = 0
        min_dist_to_any_victim = 9999.0
        
        # Victim Targeting Logic
        for v in 1:num_victims
            if victim_statuses[v] == "SEARCHING" && rover_states[active_rover] == "EXPLORING"
                dist = norm(rover_positions[active_rover] - victim_coords[v])
                if dist < min_dist_to_any_victim
                    min_dist_to_any_victim = dist
                    active_victim_idx = v
                end
            elseif assigned_rescuer[v] == active_rover
                active_victim_idx = v
                min_dist_to_any_victim = norm(rover_positions[active_rover] - victim_coords[v])
                break 
            end
        end

        # FSM Updates
        if active_victim_idx > 0 && rover_states[active_rover] == "EXPLORING"
            current_target = victim_coords[active_victim_idx]
            if min_dist_to_any_victim < cv_vision_radius
                victim_statuses[active_victim_idx] = "FOUND"
                assigned_rescuer[active_victim_idx] = active_rover
                rover_states[active_rover] = "DESCENDING"
            end
        elseif active_victim_idx > 0 && assigned_rescuer[active_victim_idx] == active_rover
            current_target = victim_coords[active_victim_idx]
            if rover_states[active_rover] == "DESCENDING"
                rover_hover_alts[active_rover] -= 0.5 
                if rover_hover_alts[active_rover] <= 0.0
                    rover_hover_alts[active_rover] = 0.0; rover_states[active_rover] = "RESCUING"
                end
            elseif rover_states[active_rover] == "RESCUING"
                victim_statuses[active_victim_idx] = "SECURED"; rover_states[active_rover] = "ASCENDING"
                setprop!(vis["environment"]["victim_$active_victim_idx"], "visible", false) 
            elseif rover_states[active_rover] == "ASCENDING"
                rover_hover_alts[active_rover] += 0.5 
                if rover_hover_alts[active_rover] >= 2.0
                    rover_hover_alts[active_rover] = 2.0; rover_states[active_rover] = "MEDICAL_EVAC"
                end
            end
        end
        
        # Base Station Routing
        if rover_states[active_rover] in ["RETURNING", "MEDICAL_EVAC"]
            current_target = base_station_coord
            dist_to_base = norm(rover_positions[active_rover] - base_station_coord)
            
            if base_station_status == "UNVERIFIED" && dist_to_base < 3.0
                global base_station_status = "DESTROYED"; global base_station_coord = secondary_base_coord
                println("   🚨 [MAYDAY] PRIMARY BASE DESTROYED! Rerouting swarm to Backup LZ.")
                setobject!(vis["environment"]["base"], Cylinder(Point3f(0,0,0), Point3f(0,0,0.2), 0.8f0), MeshLambertMaterial(color=colorant"red"))
                sec_z = clean_terrain[searchsortedfirst(x_coords, secondary_base_coord[1]), searchsortedfirst(y_coords, secondary_base_coord[2])]
                setobject!(vis["environment"]["secondary_base"], Cylinder(Point3f(0,0,0), Point3f(0,0,0.2), 0.8f0), MeshLambertMaterial(color=colorant"yellow"))
                settransform!(vis["environment"]["secondary_base"], Translation(secondary_base_coord[1] - vis_offset, secondary_base_coord[2] - vis_offset, get_visual_z(sec_z)))
                current_target = base_station_coord
            end
            
            # Drop off logic (Only trigger if base is safe!)
            if dist_to_base < 0.5 && base_station_status != "UNVERIFIED"
                if rover_states[active_rover] == "MEDICAL_EVAC"
                    println("   🏥 [EVACUATED] Drone $active_rover dropped Casualty $(active_victim_idx) at Safe Base.")
                    victim_statuses[active_victim_idx] = "EVACUATED"; assigned_rescuer[active_victim_idx] = 0
                end
                
                # 🛬 THE FIX: Don't instantly charge. Start landing!
                if rover_states[active_rover] in ["RETURNING", "MEDICAL_EVAC"]
                    rover_states[active_rover] = "LANDING"
                end
            end
        end

        # Battery Drain Threshold
        if rover_batteries[active_rover] < 25.0 && rover_states[active_rover] == "EXPLORING"
            rover_states[active_rover] = "RETURNING"
        end

        #  updated sequences
        if rover_states[active_rover] == "LANDING"
            rover_hover_alts[active_rover] -= 0.5 
            if rover_hover_alts[active_rover] <= 0.0
                rover_hover_alts[active_rover] = 0.0
                rover_states[active_rover] = "CHARGING"
            end
        elseif rover_states[active_rover] == "CHARGING"
            rover_batteries[active_rover] = min(rover_batteries[active_rover] + 20.0, 100.0)
            if rover_batteries[active_rover] == 100.0 
                rover_states[active_rover] = "LIFTING_OFF" 
            end
        elseif rover_states[active_rover] == "LIFTING_OFF"
            rover_hover_alts[active_rover] += 0.5
            if rover_hover_alts[active_rover] >= 2.0
                rover_hover_alts[active_rover] = 2.0
                rover_states[active_rover] = "EXPLORING"
            end
        end

        if rover_batteries[active_rover] < 25.0 && rover_states[active_rover] == "EXPLORING"
            rover_states[active_rover] = "RETURNING"
        end
        if rover_states[active_rover] == "CHARGING"
            rover_batteries[active_rover] = min(rover_batteries[active_rover] + 20.0, 100.0)
            if rover_batteries[active_rover] == 100.0 rover_states[active_rover] = "EXPLORING" end
        end

        # ---------------------------------------------------------
        # 🧠 ADVANCED PATH PLANNING: ARTIFICIAL POTENTIAL FIELDS (APF)
        # ---------------------------------------------------------
        is_vert = rover_states[active_rover] in ["DESCENDING", "RESCUING", "ASCENDING", "CHARGING", "LANDING", "LIFTING_OFF"]
        
        if !is_vert
            # attractive force towards target
            vector_to_target = current_target - rover_positions[active_rover]
            dist_to_target = norm(vector_to_target)
            F_att = dist_to_target > 0.1 ? normalize(vector_to_target) * 2.5 : [0.0, 0.0]
            
            # Designed a simplified swarm repulsion to prevent collisions 
            # without heavy computational overhead of full APF with obstacle mapping
            F_rep = [0.0, 0.0]
            for other_rover in 1:num_rovers
                if other_rover != active_rover && rover_states[other_rover] != "CHARGING"
                    swarm_vector = rover_positions[active_rover] - rover_positions[other_rover]
                    swarm_dist = norm(swarm_vector)
                    if swarm_dist < 3.0 && swarm_dist > 0.01 # 3-meter safety bubble
                        F_rep += normalize(swarm_vector) * (1.5 / swarm_dist^2) # Inverse square push
                    end
                end
            end
            
            # Adding a local hazard repulsion field based on the hazard map (e.g., floodwater) to encourage safer paths
            # with 3-m based LIDAR
            F_hazard = [0.0, 0.0]
            
            # Finding the drone's current grid index
            rx_idx = clamp(searchsortedfirst(x_coords, rover_positions[active_rover][1]), 1, new_x)
            ry_idx = clamp(searchsortedfirst(y_coords, rover_positions[active_rover][2]), 1, new_y)
            
            # Scanning a small local window (approx 2.5 meters around the drone)
            scan_radius = 2 
            for dx in -scan_radius:scan_radius
                for dy in -scan_radius:scan_radius
                    nx = clamp(rx_idx + dx, 1, new_x)
                    ny = clamp(ry_idx + dy, 1, new_y)
                    
                    # If this specific cell has a hazard (like floodwater)
                    if hazard_map[nx, ny] > 0.0
                        hazard_pos = [x_coords[nx], y_coords[ny]]
                        hazard_vector = rover_positions[active_rover] - hazard_pos
                        hazard_dist = norm(hazard_vector)
                        
                        # Applying a massive repulsive push away from the water
                        if hazard_dist < 2.5 && hazard_dist > 0.01
                            F_hazard += normalize(hazard_vector) * (5.0 / hazard_dist^2)
                        end
                    end
                end
            end

            #  Simplified kinematic integration (F = ma)
            total_force = F_att + F_rep
            rover_velocities[active_rover] += total_force * dt
            
            # Speed Limiter
            current_speed = norm(rover_velocities[active_rover])
            if current_speed > max_speed
                rover_velocities[active_rover] = (rover_velocities[active_rover] / current_speed) * max_speed
            end
            
            # Applying Velocity to Position
            rover_positions[active_rover] += rover_velocities[active_rover] * dt
            
            # Draining Battery based on task and movement
            hz = clean_terrain[clamp(searchsortedfirst(x_coords, rover_positions[active_rover][1]), 1, new_x), clamp(searchsortedfirst(y_coords, rover_positions[active_rover][2]), 1, new_y)]
            rover_batteries[active_rover] -= current_speed * dt * (1.5 + (((hz - min_alt) / (max_alt - min_alt)) * 3.0) + wind_penalty)
        else
            rover_velocities[active_rover] = [0.0, 0.0] # Halt horizontal momentum
        end

        # Rendering updates of states and positions of drone rovers
        hz = clean_terrain[clamp(searchsortedfirst(x_coords, rover_positions[active_rover][1]), 1, new_x), clamp(searchsortedfirst(y_coords, rover_positions[active_rover][2]), 1, new_y)]
        settransform!(vis["rovers"]["rover_$active_rover"], compose(Translation(rover_positions[active_rover][1] - vis_offset, rover_positions[active_rover][2] - vis_offset, get_visual_z(hz) + rover_hover_alts[active_rover]), LinearMap(UniformScaling(0.001))))
        
        if rover_states[active_rover] != "CHARGING"
            icon = "🚁"
            if rover_states[active_rover] == "MEDICAL_EVAC" icon = "🚑"
            elseif rover_states[active_rover] in ["DESCENDING", "ASCENDING", "RESCUING"] icon = "🧗"
            elseif rover_states[active_rover] in ["LANDING", "LIFTING_OFF"] icon = "🛗" # New Elevator Icon!
            end
            
            println("   $icon Drone $active_rover | State: $(rover_states[active_rover]) | Spd: $(round(norm(rover_velocities[active_rover]), digits=1))m/s")
        end
    end
    sleep(0.3)
end