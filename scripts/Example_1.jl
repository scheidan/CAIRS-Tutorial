## =======================================================
## Continuous Assimilation of Integrating Rain Sensors (CAIRS)
##
## Description: Example 1 - artificial data
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

    sigma = 0.1      # measurement uncertainty

     ## log of normal density, p(S|R)
    logpdf(Normal(mu, sigma), S)
end

sensor_gauge = Sensor(obs_model_gauge)


## --- define binary rain gauge

function obs_model_gauge_binary(S::String, R::Vector)

    threshold = 0.5

    if R[1] >= threshold     # "wet" condition
       prob_wet = 0.98       # 2% false negatives
    else
       prob_wet = 0.10       # 10% false positives
    end

    sigma = 0.1      # measurement uncertainty

    ## log of Bernoulli density, p(S|R)
    logpdf(Bernoulli(prob_wet), int(S=="wet"))
end

sensor_gauge_binary = Sensor(obs_model_gauge_binary)


## --- define microwave link of length 6

function obs_model_MWL(S::Float64, I::Float64)

    R_mean = I/6.0
    sigma = 0.1

    ## log of normal density, p(S|I)
    logpdf(Normal(R_mean, sigma), S)  # again, it does not have to be Gaussian
end

sensor_MWL = Sensor(obs_model_MWL, Coor(6, 0, 0)) # integrates along a path of length 6


## show the properties of all defined sensors
show(sensor_gauge)
show(sensor_gauge_binary)
show(sensor_MWL)



## ---------------------------------
## 2) Define prior


mean_GP = mean_constant(mean=0.0)

cov_GP = cov_exponential(sigma=10.0,            # standard deviation of GP
                         l_spatial=1.5,         # spatial correlation length
                         l_temporal=Minute(15), # temporal correlation length
                         gamma=1.0)             # exponent for smoothness. Must be in [0, 2].



## ---------------------------------
## 3) signals

## -- rain gauge signals
s1 = Signal(10.0, sensor_gauge, Coor(2, 6, DateTime(2014, 10, 30, 13, 00, 00)))
s2 = Signal(6.5, sensor_gauge, Coor(3.5, 1.1, DateTime(2014, 10, 30, 13, 00, 25)))

## -- binary signals
s3 = Signal("wet", sensor_gauge_binary, Coor(1.5, 0.6, DateTime(2014, 10, 30, 12, 59, 40)))
s4 = Signal("wet", sensor_gauge_binary, Coor(5.8, 3.8, DateTime(2014, 10, 30, 13, 00, 00)))
s5 = Signal("dry", sensor_gauge_binary, Coor(0.5, 2.8, DateTime(2014, 10, 30, 13, 00, 05)))

## -- MWL signals
s6 = Signal(2.1, sensor_MWL, Coor(0.5, 0.8, DateTime(2014, 10, 30, 12, 58, 58)), 0.4) # angle is in [rad]!
s7 = Signal(1.3, sensor_MWL, Coor(1.4, 3.1, DateTime(2014, 10, 30, 13, 00, 33)), 0.15)
s8 = Signal(5.2, sensor_MWL, Coor(2.1234, 7.5, DateTime(2014, 10, 30, 13, 01, 02)), -1.4)

## Vector with all signals
#signals = [s1, s2, s3, s4, s5, s6, s7, s8] # the order does not matter

signals = [s1, s2, s3, s4, s5, s6, s7, s8] # the order does not matter

## write sensor position in a file (for plotting)
sensor2csv(signals, "sensor_coor.csv")



## -----------
## 4) Define location for predictions

## create a regular grid of 400 points
nn = 20
loc_pred = [Coor(i, j, DateTime(2014, 10, 30, 13, 00, 00))
              for i=linspace(0, 8, nn), j=linspace(0, 8, nn)]

## -----------
## 5) Assimilation and interpolation

R_pred = predict(loc_pred,               # vector or array with locations for predictions
                 signals,                # vector of signals
                 mean_GP,                # mean function of prior
                 cov_GP,                 # covariance function of prior
                 n_sample_calib = 20000, # number of iterations of the Gibbs sampler
                 burn_in = 5000,         # number of removed samples (and length of adaptation)
                 n_sample_pred = 6000,   # number of samples for predictions
                 delta = Minute(3))      # consider all signals within time 'delta'
                                         #   from prediction points

## compute summary of samples and save in file
summary2csv(R_pred, "result_cairs.csv")



## -----------
## 6) Visualize the result with R

## One possibility to visualize the result is to use R. A simple
## R-script comes with the package.  R (http://www.r-project.org/) and
## the R-libraries 'lattice', 'latticeExtra' and 'tripack' must be
## installed.

output_file = "rain_map.pdf"

## Path to the R script shipped with CAIRS
pathRscript = joinpath(Pkg.dir("CAIRS"), "R", "compute_rain_map.r")

## You have to adapt the path to the R executable!
run(`'C:\Program Files\R\R-3.1.1\bin\Rscript' $pathRscript result_cairs.csv sensor_coor.csv $output_file`)


## -----------
