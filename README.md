# Optimising build order for cultural point (CP) efficiency in Travian

This is a script for generating a build order that optimise the efficiency of CP-generating buildings either by resource or time in Travian: Legends.

## Problem description

Each building generates a certain amount of CP per day. However, the cost, time to build and the CP amount is different for different buildings. This means that the CP efficiency of a building is not constant. Intuitively, we want to invest the least amount of resources to get the largest amount of CP or the least amount of time to build if we want to maximise with respect to time. Furthermore, each building has its own prerequisites buildings that include the previous level of itself or other buildings. This means that even if a higher level building is more efficient, it may not be efficient once its prerequisites are taken into account.

The goal here is to find the optimal build order from a certain starting point (current levels of all the buildings) to maximise the CP efficiency in terms of resources or time.

## Usage

1. Change the *current_level* constant in *data_processing.py* to the current level of your buildings.  
For presets, you can use "only core" or "all empty" which correspond to only major CP buildings or all buildings, respectively.

2. Run *data_processing.py* to generate the build order CSV.

## Note

- The data may not be accurate, especially for the time data.
- This is for Travian: Legends, not Travian Kingdoms.
