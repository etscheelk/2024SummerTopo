using Distances
using Plots
using Ripserer

function noisy_circle(n; r=1, noise=0.1)
    points = NTuple{2,Float64}[]
    for _ in 1:n
        theta = 2π * rand()
        push!(points, (r * sin(theta) + noise * rand(), r * cos(theta) + noise * rand()))
    end
    return points
end

circ_100 = noisy_circle(100)
scatter(circ_100; aspect_ratio=1, legend=false, title="Noisy Circle")

out = ripserer(circ_100; dim_max=3)

plot(out)   
barcode(out)

plot(out[2])
barcode(out[2:end]; linewidth=2)