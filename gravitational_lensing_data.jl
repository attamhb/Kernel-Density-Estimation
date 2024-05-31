using Revise
using Random

#using Pkg
#Pkg.add("CSV")
#Pkg.add("DataFrames")

using CSV
using DataFrames

# Create a 2D array (example data)


# Save the DataFrame to a CSV file
Random.seed!(50)

# generate data points
N = 1000
U = rand(N)
theta = 2*pi*rand(N)
R = 3*U./(5 .- 3*U)
X = [R.*cos.(theta) R.*sin.(theta)]

df = DataFrame(X, :auto)  # :auto assigns column names automatically
CSV.write("data_gravitational_lensing.csv", df)

h = 0.01                       # meshgrid h=dx=dy
dt = .1
nSteps = 2000
rGrid = 0:.01:1.7
mySpan = .005

