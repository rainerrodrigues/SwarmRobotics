# geo_mapping.jl
using Pkg
Pkg.activate(".")
using Rasters, RasterDataSources, ArchGDAL
using MeshCat, GeometryBasics, Colors, CoordinateTransformations, FileIO, MeshIO
using AbstractGPs, KernelFunctions, Random, LinearAlgebra, Statistics

# 🧠 INJECT THE SCENARIO MODULE
include(joinpath(@__DIR__, "Scenarios", "config.jl"))

# CHOOSE YOUR DISASTER HERE:
# Options: "SINGLE_VICTIM_CLEAR", "MULTIPLE_SCATTERED", "MASS_CASUALTY_WINDY", "RESCUE_HEAVY_RAIN", "CYCLONE_FLOOD"
ACTIVE_SCENARIO = "CYCLONE_FLOOD"

victim_coords, wind_penalty, cv_vision_radius, flood_level = load_scenario(ACTIVE_SCENARIO)
num_victims = length(victim_coords)

# ---------------------------------------------------------
# 1. FETCH PHYSICAL TERRAIN (Pune, Maharashtra)
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
# 2. INITIALIZE SWARM
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
rover_velocities = fill(0.0, num_rovers)     
rover_hover_alts = fill(2.0, num_rovers) 

max_speed = 3.0
acceleration = 1.0 
commercial_airspace_ceiling = 620.0 

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
# 3. MESHCAT VISUALIZER
# ---------------------------------------------------------
vis = Visualizer()
println("\n🚀 OPEN 3D ENVIRONMENT: ", vis)
sleep(5) 

setprop!(vis["/Background"], "top_color", RGB(0.05, 0.05, 0.08))
setprop!(vis["/Background"], "bottom_color", RGB(0.05, 0.05, 0.08))

function get_visual_z(real_z) return (real_z - min_alt) / (max_alt - min_alt) * 4.0 end

# Draw terrain & Base
points = Point3f[]; colors = RGBA{Float32}[]
for i in 1:new_x, j in 1:new_y
    push!(points, Point3f(x_coords[i], y_coords[j], get_visual_z(clean_terrain[i, j])))
    nz = Float32((clean_terrain[i, j] - min_alt) / (max_alt - min_alt))
    r = nz > 0.6 ? nz : 0.2f0 + nz; g = nz > 0.6 ? nz : 0.6f0 - (nz * 0.5f0); b = nz > 0.6 ? nz : 0.1f0
    push!(colors, RGBA{Float32}(r, g, b, 1.0f0))
end
setobject!(vis["environment"]["ghats"], PointCloud(points, colors), PointsMaterial(size=0.15))
setobject!(vis["environment"]["base"], Cylinder(Point3f(0,0,0), Point3f(0,0,0.2), 0.8f0), MeshLambertMaterial(color=colorant"yellow"))
settransform!(vis["environment"]["base"], Translation(base_station_coord[1], base_station_coord[2], get_visual_z(base_station_z)))

# 🌊 RENDER FLOODWATER IF ACTIVE
hazard_map = zeros(new_x, new_y)
if flood_level > 0.0
    flood_points = Point3f[]; flood_colors = RGBA{Float32}[]
    threshold = min_alt + flood_level
    for i in 1:new_x, j in 1:new_y
        if clean_terrain[i,j] < threshold
            hazard_map[i, j] = 50.0 # Mathematical wall
            push!(flood_points, Point3f(x_coords[i], y_coords[j], get_visual_z(clean_terrain[i,j]) + 0.1))
            push!(flood_colors, RGBA{Float32}(0.0f0, 0.4f0, 0.8f0, 0.7f0))
        end
    end
    setobject!(vis["environment"]["flood"], PointCloud(flood_points, flood_colors), PointsMaterial(size=0.18))
end

# 🆘 SPAWN MULTIPLE VICTIMS
victim_statuses = fill("SEARCHING", num_victims)
assigned_rescuer = fill(0, num_victims)

for (idx, coord) in enumerate(victim_coords)
    hx = searchsortedfirst(x_coords, coord[1])
    hy = searchsortedfirst(y_coords, coord[2])
    hz = clean_terrain[hx, hy]
    setobject!(vis["environment"]["victim_$idx"], Sphere(Point3f(0,0,0), 0.15f0), MeshLambertMaterial(color=colorant"orange"))
    settransform!(vis["environment"]["victim_$idx"], Translation(coord[1], coord[2], get_visual_z(hz)))
end

drone_mesh = try load(joinpath(@__DIR__, "new_dronev_whole.stl")) catch; Rect3f(Vec3f(-0.3, -0.3, -0.3), Vec3f(0.6, 0.6, 0.6)) end
for r in 1:num_rovers setobject!(vis["rovers"]["rover_$r"], drone_mesh, MeshLambertMaterial(color=colorant"red")) end

# ---------------------------------------------------------
# 4. MULTI-VICTIM FSM
# ---------------------------------------------------------
println("\n=================================================")
println("🚀 S.A.R. FLEET LAUNCHED")
println("=================================================")

for step in 1:80 
    println("\n--- Cycle $step ---")
    p_fx = posterior(f(X_obs, 0.05), y_obs)
    uncertainty = reshape(var.(marginals(p_fx(grid_points))), new_x, new_y)
    
    utility = zeros(new_x, new_y)
    Threads.@threads for i in 1:new_x
        for j in 1:new_y
            actual_alt = clean_terrain[i,j]
            utility[i, j] = uncertainty[i, j] - (0.8 * ((actual_alt - min_alt) / (max_alt - min_alt))) - hazard_map[i, j] - (actual_alt > commercial_airspace_ceiling ? 9999.0 : 0.0)
        end
    end
    _, max_idx = findmax(utility)
    ideal_target = [x_coords[max_idx[1]], y_coords[max_idx[2]]]
    
    for active_rover in 1:num_rovers
        current_target = ideal_target
        
        # 1. FIND CLOSEST UNRESCUED VICTIM
        active_victim_idx = 0
        min_dist_to_any_victim = 9999.0
        
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
                break # Locked onto this specific casualty
            end
        end

        # 2. CV & RESCUE LOGIC
        if active_victim_idx > 0 && rover_states[active_rover] == "EXPLORING"
            current_target = victim_coords[active_victim_idx]
            if min_dist_to_any_victim < cv_vision_radius
                println("   📷 [VISION] Drone $active_rover locked onto Casualty $(active_victim_idx)!")
                victim_statuses[active_victim_idx] = "FOUND"
                assigned_rescuer[active_victim_idx] = active_rover
                rover_states[active_rover] = "DESCENDING"
                rover_velocities[active_rover] = 0.0 
            end
            
        elseif active_victim_idx > 0 && assigned_rescuer[active_victim_idx] == active_rover
            current_target = victim_coords[active_victim_idx]
            if rover_states[active_rover] == "DESCENDING"
                rover_hover_alts[active_rover] -= 0.5 
                if rover_hover_alts[active_rover] <= 0.0
                    rover_hover_alts[active_rover] = 0.0
                    rover_states[active_rover] = "RESCUING"
                end
            elseif rover_states[active_rover] == "RESCUING"
                victim_statuses[active_victim_idx] = "SECURED"
                rover_states[active_rover] = "ASCENDING"
                setprop!(vis["environment"]["victim_$active_victim_idx"], "visible", false) 
                println("   ✅ [SECURED] Drone $active_rover winched Casualty $(active_victim_idx). Ascending.")
            elseif rover_states[active_rover] == "ASCENDING"
                rover_hover_alts[active_rover] += 0.5 
                if rover_hover_alts[active_rover] >= 2.0
                    rover_hover_alts[active_rover] = 2.0
                    rover_states[active_rover] = "MEDICAL_EVAC"
                end
            elseif rover_states[active_rover] == "MEDICAL_EVAC"
                current_target = base_station_coord
                if norm(rover_positions[active_rover] - base_station_coord) < 0.5
                    println("   🏥 [EVACUATED] Drone $active_rover dropped Casualty $(active_victim_idx) at Base.")
                    victim_statuses[active_victim_idx] = "EVACUATED"
                    assigned_rescuer[active_victim_idx] = 0
                    rover_states[active_rover] = "CHARGING"
                    rover_velocities[active_rover] = 0.0
                end
            end
            
        elseif rover_states[active_rover] == "RETURNING" || rover_states[active_rover] == "CHARGING"
            current_target = base_station_coord
            if norm(rover_positions[active_rover] - base_station_coord) < 0.5
                rover_states[active_rover] = "CHARGING"
                rover_velocities[active_rover] = 0.0
                rover_batteries[active_rover] += 20.0
                if rover_batteries[active_rover] >= 100.0
                    rover_batteries[active_rover] = 100.0
                    rover_states[active_rover] = "EXPLORING"
                end
            end
        end

        if rover_batteries[active_rover] < 25.0 && rover_states[active_rover] == "EXPLORING"
            rover_states[active_rover] = "RETURNING"
            println("   ⚠️ Drone $active_rover Bingo Fuel. RTB.")
        end

        # 3. KINEMATICS & WEATHER PHYSICS
        is_vert = rover_states[active_rover] in ["DESCENDING", "RESCUING", "ASCENDING", "CHARGING"]
        dist_to_target = norm(rover_positions[active_rover] - current_target)

        if !is_vert
            rover_velocities[active_rover] = dist_to_target > 0.1 ? min(rover_velocities[active_rover] + acceleration, max_speed) : 0.0 
            actual_dist = min(dist_to_target, rover_velocities[active_rover])
            
            if actual_dist > 0
                rover_positions[active_rover] += (normalize(current_target - rover_positions[active_rover]) * actual_dist)
                hz = clean_terrain[clamp(searchsortedfirst(x_coords, rover_positions[active_rover][1]), 1, new_x), clamp(searchsortedfirst(y_coords, rover_positions[active_rover][2]), 1, new_y)]
                
                # 🌩️ THE WEATHER DRAIN: Steepness + Wind Penalty
                rover_batteries[active_rover] -= actual_dist * (1.5 + (((hz - min_alt) / (max_alt - min_alt)) * 3.0) + wind_penalty)
            end
        end

        # 4. RENDER
        hz = clean_terrain[clamp(searchsortedfirst(x_coords, rover_positions[active_rover][1]), 1, new_x), clamp(searchsortedfirst(y_coords, rover_positions[active_rover][2]), 1, new_y)]
        settransform!(vis["rovers"]["rover_$active_rover"], compose(Translation(rover_positions[active_rover][1], rover_positions[active_rover][2], get_visual_z(hz) + rover_hover_alts[active_rover]), LinearMap(UniformScaling(0.001))))
        
        if rover_states[active_rover] != "CHARGING"
            icon = rover_states[active_rover] == "MEDICAL_EVAC" ? "🚑" : (rover_states[active_rover] in ["DESCENDING", "ASCENDING", "RESCUING"] ? "🧗" : "🚁")
            println("   $icon Drone $active_rover | State: $(rover_states[active_rover]) | Batt: $(round(rover_batteries[active_rover], digits=1))%")
        end
    end
    sleep(0.3)
end