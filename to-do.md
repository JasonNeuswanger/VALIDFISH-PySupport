# Drift foraging model to-do list

## Visualization

* Do a volumetric plot of maneuver energy expenditure in 3D, weighted by maneuver frequency in each location

* Do a plot of mean detection position in the x plane from a top view and separately from a side view, with two lines for each prey type, a solid one at the outer limit and a dotted one at the mean position.

# Tomorrow

* Build a single function to do all the detection proportions
  without all the redundant recalculation of overall totals.
  
* Automate PDF exports for all the new plots.

## Scientific details

### Optimization

* Test wolf count vs iteration tradeoff

### Major tasks

* Plotting functions & reality checks

### Minor adjustments

* Might need to make Z_0 a function of fish size for the same parameters to work across sizes
* Make discriminability a function of prey feature size

## Diagnostics

* Add blanket multipliers on prey/debris density just to make it possible to investigate this variable as a parameter (normally set to 1 / not optimized).

### Calibration/optimization/testing

Here's how this works.

1. We want to find the values of the fish's decision variables that maximize its NREI. These variables include water velocity, angular width of the search volume (theta radians), radius of the search volume, saccade_time, discrimination threshold (lambda_c), and attentional allocation (alpha_i). That's 5+n_prey_categories variables to be optimized by each fish in each unique situation.

2. Based on the assumption that fish behave optimally and the optimum calculated behavior, we need to generate a suite of metrics that can be tested against real data. Also, we need a way to summarize all these fit metrics into a single summary statistic we can use to determine model fit (weighted sum of squares, with weights based on the sensitivity of NREI to each variable?). 
These include:
    * Actual focal velocity.
    * Foraging attempt rate.
    * Proportion of rejections among foraging attempts.
    * For each prey category except the last (N-1 df), the proportion of the diet it represents.
    * The spatial distribution of foraging attempts.
    
3. Finally, we need to calculate the distributions of unknown parameters based on how well our real fish fit the data. These include 8+n_prey_categories variables to be estimated by ABC:
    * The crypticity of all N prey categories
    * delta_0, the scaling parameter for the effect of angular size on detection
    * t_V, scaling parameter for the effect of search volume V on search rate Z
    * Z_0, scaling parameter for the effect of search rate on detection probability
    * beta, scaling parameter for the effect of saccade time on detection probability
    * alpha_0, scaling parameter "" feature attention allocation ""
    * sigma_t, standard dev of preyishness of prey of all types
    * c_1, parameter scaling effects of feature-based attention and saccade time on uncertainty
    * lambda_d, difference in preyisness between prey and debris, assumed constant across types
  
4. We generate predictions for each real fish in the "test" set, rather than the calibration set, based on many different draws from the distributions of the parameters from step 3.

### Low-priority tweaks and polishing

* Consider average rotation of prey into angular size calculations... although the effect of that might be rolled into the constant governing the effect of angular size, anyway, and it might therefore be unnecessary to modify. The other difficulty here is that some prey are more round and some are more elongated, so a single formula wouldn't work.
* Maneuver costs are based on the water velocity at the focal point. They should be based on the average water velocity mid-way between the prey and the focal point. Building this into the model would be a nightmare because it would require using a different 
 interpolating function from a different csv within the integrand of an integral.
 Instead, we should assume some constant velocity, but probably choose it based on 
 some criterion that puts it higher than the focal velocity. This is going to be an 
 overall scaling factor on energy costs so it's important to choose something plausible.
* Look into the effects of edge behavior in the velocity function at the water's surface. It asymptotically went to infinity at the surface when I used the original function, so I dialed it down a hair to avoid that infinity by making the real "surface" from velocity profile standpoint be slightly above the actual surface. However, there is probably still an unrealistic boost to velocity right at/below the surface, which might make a difference in results, and could be remedied by modifying the function.
* Account for proportion of energy assimilated.
* Update prey mass/energy content calculations to latest from empirical fish analysis.

## Speed improvements

### General

* Test quadpack version of integrate_over_cross_section vs simpsons verson
* Start doing some code profiling to figure out priority areas and avoid over-complicating things with unnecessary optimizations, especially compilation. This should be with
  caching enabled, but doing realistically complete tasks (i.e. NREI calculations) that 
  don't allow caching to gloss over some expensive stuff, forcing caches to rebuild/grow
  as they would in a real application.
* Work on figuring out how to make the integrations finish faster and seek less precision.

### Memozing with lru_cache

* ROUNDING -- round certain things to nearest 0.01 or 0.001 so the cache usually hits
* Make Maxsize for the caches a power of 2 (docs say it performs best then)
* Figure out sensible cache sizes for each function.
* Can use maxsize=None for some, along with cache_clear() to flush the cache from memory when it's no longer in use... this might be specially important for NREI where the numerator and denominator are replicating expensive calculations: https://stackoverflow.com/questions/37653784/how-do-i-use-cache-clear-on-python-functools-lru-cache

### Compilation with numba

* Maybe put calculation of temporal integration bounds into a jitclass.
* See if we can do "maneuver cost at mean position" as a proxy for "mean maneuver cost"...
  and how much of a difference that makes in predicted optima.
* Figure out how to use dot product in a jitted function for mg.maneuver_2D_coords
* Make prey angular size a jitted function.
* Possibly make functions in the integrand of minor_spherical_sector_volume into externally-defined, jitted functions.
* Make PreyCategory and ForagingEnvironment both jitclasses.
* Could Jit the steady swimming cost model functions in the brett_glass_parameters module, but memoziation is probably sufficient.

## Code cleanup

* Make a subclass like PlottableFish and put plotting/diagnostic functions there.
* Reduce Python 2 relic floats with integers.
* Make functions properties where appropriate.

# Implemented implifications
* Assume angular size is constant (based on some average), so tau doesn't depend
  on t, and detection probability has a closed-form solution.
* Assume all maneuvers are based on detection on the upstream edge of the reaction volume.

# Possible simplifications

* Could calculate search rate from fixed focal velocity rather than integrating velocity over vertical profile.
* Build some default rules into prey category handling, such as always allocate
  zero attention to prey classes with 0 drift density or to mites, because it will
  always be optimal to ignore them.
* Ignore vertical variation in water velocity
* Reduce dimensionality of prey category strategies by changing to a single dimension reflecting various strategies instead of a separate dimension for each prey categorys. Strategies include equal attention, all attention to biggest prey, all attention to most abundant prey, attention proportionate to size and/or abundance.
  
# Debugging

* Make sure detection probabilities are never coming back negative, or figure out why they are.
* At some point I need to think about how to handle items that show up in the diet but not the drift, for testing purposes. Currently any category not include in the drift is excluded from the model altogether.