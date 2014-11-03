## =======================================================
## Continuous Assimilation of Integrating Rain Sensors (CAIRS)
##
## Description: Example 2 - measured data
##
## Andreas Scheidegger -- andreas.scheidegger@eawag.ch
## =======================================================


using CAIRS
using Dates
using Distributions


## ---------------------------------
## 1) Define sensors

## --- define rain gauge

function obs_model_gauge(S::Float64, R::Vector)

    mu = R[1]        # Note, the signal and could be non-linearly
                     # related to the rain intensity.

     # 15% relative error, but at least 1
    sigma = max(1, 0.15*S)

     ## log of normal density, p(S|R)
    logpdf(Normal(mu, sigma), S)
end

sensor_gauge = Sensor(obs_model_gauge)


## --- define microwave link with different lengths
##
## (Obviously, one should define a little helper function if more than three MWLs must be defined)

## MWL 3.016 km long
function log_p_MWL_3016(S::Real, I::Float64)

    sigma = max(1, 0.3*S)               # 30% realtive error
    R_mean = I/3016

    ## log of normal density, p(S|I)
    logpdf(Normal(R_mean, sigma), S)
end
sensor_MWL_3016 = Sensor(log_p_MWL_3016, Coor(3016, 0.0, 0.0)) # integrates along a path

## MWL 2.174 km long
function log_p_MWL_2174(S::Real, I::Float64)

    sigma = max(1, 0.3*S)               # 30% realtive error
    R_mean = I/2174

    ## log of normal density, p(S|I)
    logpdf(Normal(R_mean, sigma), S)
end
sensor_MWL_2174 = Sensor(log_p_MWL_2174, Coor(2174, 0.0, 0.0)) # integrates along a path

## MWL 6.316 km long
function log_p_MWL_6316(S::Real, I::Float64)

    sigma = max(1, 0.3*S)               # 30% realtive error
    R_mean = I/6316

    ## log of normal density, p(S|I)
    logpdf(Normal(R_mean, sigma), S)
end
sensor_MWL_6316 = Sensor(log_p_MWL_6316, Coor(6316, 0.0, 0.0)) # integrates along a path



## ---------------------------------
## 2) Define prior


mean_GP = mean_constant(mean=0.0)

cov_GP = cov_exponential(sigma=10.0,            # standard deviation of GP
                         l_spatial=3000.0,      # spatial correlation length [m]
                         l_temporal=Minute(5),  # temporal correlation length
                         gamma=1.0)             # exponent for smoothness. Must be in [0, 2].



## ---------------------------------
## 3) signals

## The files must contain two columns:
##  Column 1: holds date and time
##  Column 2: holds the signal values

signals_gauges = Signal[]               # create an empty array
signals_MWL = Signal[]               # create an empty array


## Gauges signals
add_signal!(signals_gauges, "../data/Station1.csv", sensor_gauge, Coor(2085, 3498), delim=',')
add_signal!(signals_gauges, "../data/Station2.csv", sensor_gauge, Coor(2389, 2191), delim=',')
add_signal!(signals_gauges, "../data/Station3.csv", sensor_gauge, Coor(3044, 2718), delim=',')
add_signal!(signals_gauges, "../data/Station4.csv", sensor_gauge, Coor(1447, 2924), delim=',')
add_signal!(signals_gauges, "../data/Station5.csv", sensor_gauge, Coor(2125,  989), delim=',')


## MWLs
add_signal!(signals_MWL, "../data/Link1.csv", sensor_MWL_3016,
            Coor(700, 2710),   # coordinate of one antenna
            0.536667,          # rotation of the MWL around the first coordinate in [rad]
            delim=',')

add_signal!(signals_MWL, "../data/Link2.csv", sensor_MWL_2174,
            Coor(700, 2710), 1.343446,
            delim=',')

add_signal!(signals_MWL, "../data/Link3.csv", sensor_MWL_6316,
            Coor(700, 2710), -0.87744,
            delim=',')


## --- combine MWL and gauge signal sinone vector

signals = [signals_MWL, signals_gauges]
## signals = signals_gauges    # use only gauge signals
## signals = signals_MWL       # use only MWL signals


## write sensor position in a file (for plotting)
sensor2csv(signals, "sensor_coor.csv")



## -----------
## 4) Define location for predictions

## create a regular grid of 900 points
nn = 30
loc_pred = [Coor(i, j, DateTime(2013, 06, 09, 21, 38, 00))
              for i=linspace(0, 4000, nn), j=linspace(0, 5000, nn)]


## -----------
## 5) Assimilation and interpolation

R_pred = predict(loc_pred,               # vector or array with locations for predictions
                 signals,                # vector of signals
                 mean_GP,                # mean function of prior
                 cov_GP,                 # covariance function of prior
                 n_sample_calib = 20000, # number of iterations of the Gibbs sampler
                 burn_in = 5000,         # number of removed samples (and length of adaptation)
                 n_sample_pred = 6000,   # number of samples for predictions
                 delta = Second(90))     # consider all signals within time 'delta'
                                         #   from prediction points

## compute summary of samples and save in file
summary2csv(R_pred, "result_cairs.csv")



## -----------
## 6) Visualize the result with R

## One possibility to visualize the result is to use R. A simple
## R-script comes with the package.  R (http://www.r-project.org/) and
## the R-libraries 'lattice', 'latticeExtra' and 'tripack' must be
## installed.

output_file = "rain_map_2.pdf"

## Path to the R script shipped with CAIRS
pathRscript = joinpath(Pkg.dir("CAIRS"), "R", "compute_rain_map.r")

## You have to adapt the path to the R executable!
run(`'C:\Program Files\R\R-3.1.1\bin\Rscript' $pathRscript result_cairs.csv sensor_coor.csv $output_file`)


## -----------
