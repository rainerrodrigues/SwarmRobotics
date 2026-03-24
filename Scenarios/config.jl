# Scenarios/config.jl

function load_scenario(scenario_name)
    println("🌩️  LOADING SCENARIO: $scenario_name")
    
    # 1. Baseline Defaults
    victims = [[14.0, 14.0]] # Array of [X, Y] coordinates
    wind_penalty = 0.0       # Extra battery drain from fighting wind
    vision_radius = 3.0      # How close the drone needs to be to spot the victim
    flood_level = 0.0        # Altitude offset for floodwaters
    
    # 2. Scenario Definitions
    if scenario_name == "SINGLE_VICTIM_CLEAR"
        # Uses baseline defaults
        
    elseif scenario_name == "MULTIPLE_SCATTERED"
        victims = [[14.0, 14.0], [5.0, 18.0], [18.0, 5.0]]
        
    elseif scenario_name == "MASS_CASUALTY_WINDY"
        # 3 victims clustered on one peak. High wind drains battery faster.
        victims = [[14.0, 14.0], [14.2, 14.1], [13.9, 14.0]] 
        wind_penalty = 2.0 
        
    elseif scenario_name == "RESCUE_HEAVY_RAIN"
        victims = [[5.0, 18.0]]
        vision_radius = 0.8 # Rain blinds the camera; drone must practically fly over them
        
    elseif scenario_name == "CYCLONE_FLOOD"
        victims = [[18.0, 18.0], [2.0, 2.0]]
        wind_penalty = 3.5
        vision_radius = 0.5
        flood_level = 15.0 # Floods the lowest 15 meters of the map
        
    elseif scenario_name == "EARTHQUAKE"
        victims = [[10.0, 5.0], [15.0, 15.0]]
        # Earthquakes don't affect vision, but the base station might be compromised (advanced logic needed later)
    else
        println("⚠️ Unknown scenario. Defaulting to SINGLE_VICTIM_CLEAR.")
    end
    
    return victims, wind_penalty, vision_radius, flood_level
end