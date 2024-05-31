using Revise
using PyPlot
using Random
using Loess
push!(LOAD_PATH, ".")
using PIC_method
using Consensus_Delaunay

# generate points according to the following distribution
#   f(x,y) = 5/6π * 1/(r(1+r)^2)  for 0<r<1.5 with r^2=x^2+y^2
# The pdf of R=sqrt(X^2+Y^2) is given by:
#    g(r) = 5/3 * 1/(1+r)^2   for 0<r<1.5
#    G(r) = 5/3*r/(1+r)       for 0<r<1.5
#    G^-1(z) = 3z/(5-3z)      for 0<z<1

Random.seed!(50)

# generate data points
N = 1000
U = rand(N)
theta = 2*pi*rand(N)
R = 3*U./(5 .- 3*U)
X = [R.*cos.(theta) R.*sin.(theta)]
h = 0.01                       # meshgrid h=dx=dy
dt = .1
nSteps = 2000
rGrid = 0:.01:1.7
mySpan = .005


# A) initializiation
#-------------------
# final estimate
time_int, saving_X, xGrid, yGrid, saving_label, saving_density = ode_consensus_Delaunay(X,h,dt,nSteps)

function radial_function(xGrid,yGrid,f,rGrid,method,mySpan)
    # init
    x = xGrid*ones(length(yGrid))'
    y = ones(length(xGrid))*yGrid'
    r = sqrt.(x.^2 + y.^2)
    if (method=="loess")
        # lowess
        model = loess(r[:], f[:], span=mySpan)
        rho = predict(model, rGrid)
        return rGrid,rho
    else
        rMin,rMax,Δr = rGrid[1],rGrid[end],rGrid[2]-rGrid[1]
        rInt,_,rho = PIC_1D(rMin,rMax,Δr,r[:],false,f[:])
        return rInt,rho
    end
end

figure(1, figsize=(40, 20));clf();
subplot(221)
imshow(transpose(saving_label[1]), extent=(xGrid[1], xGrid[end], yGrid[1], yGrid[end]), origin="lower")
xlabel("x");ylabel("y")
scatter(saving_X[1][:, 1], saving_X[1][:, 2], color="red",s=20)
title("Voronoi cells")
subplot(222)
imshow(transpose(saving_label[101]), extent=(xGrid[1], xGrid[end], yGrid[1], yGrid[end]), origin="lower")
xlabel("x");ylabel("y")
scatter(saving_X[101][:, 1], saving_X[101][:, 2], color="red",s=20)
title("Voronoi cells")
subplot(223)
imshow(transpose(saving_label[1001]), extent=(xGrid[1], xGrid[end], yGrid[1], yGrid[end]), origin="lower")
xlabel("x");ylabel("y")
scatter(saving_X[1001][:, 1], saving_X[1001][:, 2], color="red",s=20)
title("Voronoi cells")
subplot(224)
imshow(transpose(saving_label[end]), extent=(xGrid[1], xGrid[end], yGrid[1], yGrid[end]), origin="lower")
xlabel("x");ylabel("y")
scatter(saving_X[end][:, 1], saving_X[end][:, 2], color="red",s=20)
title("Voronoi cells")

figure(2);clf()
semilogy(rGrid,(rGrid.<1.5).*5/(6*pi)*1.0./(rGrid.*(1.0.+rGrid).^2),"-.",label="analytic")
int_show = [0,100,500,1000,2000]
for k in int_show
    r,rho = radial_function(xGrid,yGrid,saving_density[k+1],rGrid,"PIC",mySpan)
    #semilogy(r[2:end],rho[2:end],label="t="*string((k-1)*dt))
    semilogy(r,rho,label="t="*string(k*dt))
end
grid();xlabel("r");ylabel("slice x");legend()


# plot all the labels
if (1==2)
    figure(1, figsize=(30, 20));clf();
    subplot(321)
    imshow(transpose(saving_label[1]), extent=(xGrid[1], xGrid[end], yGrid[1], yGrid[end]), origin="lower")
    xlabel("x");ylabel("y")
    scatter(saving_X[1][:, 1], saving_X[1][:, 2], color="red")
    title("Voronoi cells")
    subplot(322)
    rGrid,rho = radial_function(xGrid,yGrid,saving_density[1],.01,1.7,.01)
    semilogy(rGrid,rho)
    plot(rGrid,(rGrid.<1.5).*5/(6*pi)*1.0./(rGrid.*(1.0.+rGrid).^2),"-.")
    grid();xlabel("r");ylabel("slice x");

    k = 5
    subplot(323)
    imshow(transpose(saving_label[1+k]), extent=(xGrid[1], xGrid[end], yGrid[1], yGrid[end]), origin="lower")
    xlabel("x");ylabel("y")
    scatter(saving_X[1+k][:, 1], saving_X[1+k][:, 2], color="red")
    title("Voronoi cells")
    subplot(324)
    rGrid,rho = radial_function(xGrid,yGrid,saving_density[2],.01,1.7,.01)
    semilogy(rGrid,rho)
    plot(rGrid,(rGrid.<1.5).*5/(6*pi)*1.0./(rGrid.*(1.0.+rGrid).^2),"-.")
    grid();xlabel("r");ylabel("slice x");

    k = 10
    subplot(325)
    imshow(transpose(saving_label[1+k]), extent=(xGrid[1], xGrid[end], yGrid[1], yGrid[end]), origin="lower")
    xlabel("x");ylabel("y")
    scatter(saving_X[1+k][:, 1], saving_X[1+k][:, 2], color="red")
    title("Voronoi cells")
    subplot(326)
    rGrid,rho = radial_function(xGrid,yGrid,saving_density[3],.01,1.7,.01)
    semilogy(rGrid,rho)
    plot(rGrid,(rGrid.<1.5).*5/(6*pi)*1.0./(rGrid.*(1.0.+rGrid).^2),"-.")
    grid();xlabel("r");ylabel("slice x");
end
