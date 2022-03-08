using CUDA, CairoMaki

α  = 0.01                                                                            # Diffusivity
L  = 0.1                                                                             # Length
W  = 0.1                                                                             # Width
M  = 66                                                                              # No.of steps
Δx = L/(M-1)                                                                         # x-grid spacing
Δy = W/(M-1)                                                                         # y-grid spacing
Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))                                         # Largest stable time step

function diffuse!(data, a, Δt, Δx, Δy)
    dij  = view(data, 2:M-1, 2:M-1)
    di1j = view(data, 1:M-2, 2:M-1)
    dij1 = view(data, 2:M-1, 1:M-2)
    di2j = view(data, 3:M  , 2:M-1)
    dij2 = view(data, 2:M-1, 3:M  )                                                 # Stencil Computations
  
    @. dij += α * Δt * (
        (di1j - 2 * dij + di2j)/Δx^2 +
        (dij1 - 2 * dij + dij2)/Δy^2)                                               # Apply diffusion

    @. data[1, :] += α * Δt * (2*data[2, :] - 2*data[1, :])/Δx^2
    @. data[M, :] += α * Δt * (2*data[M-1, :] - 2*data[M, :])/Δx^2
    @. data[:, 1] += α * Δt * (2*data[:, 2]-2*data[:, 1])/Δy^2
    @. data[:, M] += α * Δt * (2*data[:, M-1]  -2*data[:, M])/Δy^2                  # update boundary condition (Neumann BCs)

end

domain     = zeros(M,M)                                                             # zeros Matrix 
domain_GPU = CuArray(convert(Array{Float32}, domain))                               # change the system to GPU instead of CPU
domain_GPU[32:33, 32:33] .= 5                                                       # heat Source 

heatmap(domain_GPU)

for i in 1:1000                                                                     # Apply the diffuse 1000 time to let the heat spread a long the plate       
    diffuse!(domain_GPU, α, Δt, Δx, Δy)
 if i % 20 == 0                                                                     # See the spread a long only 50 status 
     display(heatmap(domain_GPU))
 end
end
