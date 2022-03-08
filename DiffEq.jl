using CUDA 
using CairoMakie
using DifferentialEquations
using  DiffEqGPU
using Zygote

#= using the DifferentialEquations.jl packages
   to calculate the heat transfer 
   on 2-Dimesional plate, By appling 
   Neumann Boundary Condition
=#

α  = 0.01                                                  # Diffusivity
L  = 0.1                                                   # Length
W  = 0.1                                                   # Width
Nx = 66                                                    # No.of steps in x-axis
Ny = 66                                                    # No.of steps in y-axis
Δx = L/(Nx-1)                                              # x-grid spacing
Δy = W/(Ny-1)                                              # y-grid spacing
Δt = Δx^2 * Δy^2 / (2.0 * α * (Δx^2 + Δy^2))               # Largest stable time step
p = (α,Δx,Δy,100.0,0.0,100.0,0.0, Nx,Ny)                   # Parameters      
tspan =(0.0:1.0)
xspan = 0 : Δx : L




function diffuse!(dθ, θ, p, tspan)

        dθ .= α/Δx^2*(di1j - 2*dij + di2j)
           +  α/Δy^2*(dij1 - 2*dij + dij2)


               θ[1, :]    .= α*(2*θ[2, :] - 2*θ[1, :])/Δx^2
           θ[Nx, :]   .= α*(2*θ[Nx-1, :] - 2*θ[Nx, :])/Δx^2[:, 1]    .= α *(2*du[:, 2]-2*du[:, 1])/Δy^2 
    du[:, Ny]   .= α * (2*du[:, Ny-1]  -2*du[:, Ny])/Δy^2                              # update boundary condition (Dirichlet BCs) 

end

function heat_eqX!(dθ, θ, p, tspan)
        dij  = view(θ, 2:Nx-1, 2:Ny-1)
        di1j = view(θ, 1:Nx-2, 2:Ny-1)
        di2j = view(θ, 2:Nx-1, 1:Ny-2)
        dij1 = view(θ, 3:Nx  , 2:Ny-1)
        dij2 = view(θ, 2:Nx-1, 3:Ny) 
        @.  dθ = (1/Δx^2) * α * (di1j - 2*dij + di2j) + 
                 (1/Δy^2) * α * Ny * (dij1 - 2*dij + dij2)
        @. θ[1, :] = α*(2*θ[2, :] - 2*θ[1, :])/Δx^2
        @.θ[Nx, :] = α*(2*θ[Nx-1, :] - 2*θ[Nx, :])/Δx^2
    end

function heat_eqY!(dθ, θ, p, tspan)
        dij  = view(θ, 2:Nx-1, 2:Ny-1)
        di2j = view(θ, 3:Nx  , 2:Ny-1)
        dij2 = view(θ, 2:Nx-1, 3:Ny) 
        dθ .= (1/Δy^2) * α * Ny * (dij1 - 2*dij + dij2)
        @. θ[:, 1] += α  * (2*θ[:, 2]-2*θ[:, 1])/Δy^2
        @. θ[:, Ny] += α  * (2*θ[:, Ny-1]  -2*θ[:, Ny])/Δy^2 
end

function Stencil(θ ,Nx ,Ny) 
       
end 

# α,Δx,Δy=p

domain = zeros(Nx, Ny)
domain[16:32, 16:32] .= 5


prob = ODEProblem(heat_eqX!,domain,tspan)

sol = solve(prob,Tsit5(),save_everystep=true)                                  # using Runge-Kutta method 
#sol = solve(prob,Euler(),dt=Δt,progress=true, save_everystep=true, save_start=true)                                  # using Eural method  
#sol = solve(prob,CVODE_BDF(linear_solver= :GMRES),save_everystep=false)         # using the default dense Jacobian  CVODE_BDF       
                                                                                # CVODE_BDF allows us to use a sparse
                                                                                # Newton-Krylov solver by setting linear_solver = :GMRES


using Plots
gr()

plot(sol) # Plots the solution
