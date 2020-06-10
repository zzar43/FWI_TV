using Distributed, LinearAlgebra
include("acoustic_solver_parallel.jl")

# input: u
# output: y, Qy

function forward_modelling(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    source_num = size(source_position,1)
    receiver_num = size(receiver_position,1)
    data_forward = SharedArray{Float64}(Nt, receiver_num, source_num)
    wavefield_forward = SharedArray{Float64}(Nx, Ny, Nt, source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        U1, data1 = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        data_forward[:,:,ind] = data1;
        wavefield_forward[:,:,:,ind] = U1;
    end
    data_forward = Array(data_forward)
    wavefield_forward = Array(wavefield_forward)
    
    return wavefield_forward, data_forward
end


# (QDF[u])^* y_0 = u_0
# when y_0 = y_d - Qy, \nabla J(u) = u_0
# input: u, y, y_0
# output: u_0

function adjoint_op(wavefield_forward, y0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    
    c = reshape(c, Nx, Ny)
    source_num = size(source_position,1)
    u0 = SharedArray{Float64}(Nx, Ny, source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        @views adj_source = y0[end:-1:1,:,ind];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = zeros(Nx, Ny, Nt)
        @views @. utt[:,:,2:end-1] = (wavefield_forward[:,:,3:end,ind]-2*wavefield_forward[:,:,2:end-1,ind]+wavefield_forward[:,:,1:end-2,ind]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3) * dt
        u0[:,:,ind] = grad0
        
    end
    
    u0 = Array(u0)
    u0 = sum(u0, dims=3)
    u0 = u0[:,:,1]
    u0 = reshape(u0, Nx*Ny,1)
    
    return u0
end


# single scattering
function single_scattering(wavefield_forward, u0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    
    c = reshape(c, Nx, Ny)
    u0 = reshape(u0, Nx, Ny)
    source_num = size(wavefield_forward,4)
    receiver_num = size(receiver_position,1)
    data_single = SharedArray{Float64}(Nt, receiver_num, source_num)
    
    source_position_s = zeros(Int,Nx*Ny,2)
    for i = 1:Nx
        for j = 1:Ny
            source_position_s[(i-1)*Ny+j,1] = i
            source_position_s[(i-1)*Ny+j,2] = j
        end
    end
    
    @inbounds @sync @distributed for ind = 1:source_num
        
        # single scattering
        forward_wavefield_tt = zeros(Nx, Ny, Nt)
        @views @. forward_wavefield_tt[:,:,2:end-1] = (wavefield_forward[:,:,3:end,ind]-2*wavefield_forward[:,:,2:end-1,ind]+wavefield_forward[:,:,1:end-2,ind]) ./ dt^2;
        source_s = zeros(Nt, Nx*Ny)
        for i = 1:Nx
            for j = 1:Ny
                @. source_s[:,(i-1)*Ny+j] = 2*forward_wavefield_tt[i,j,:] .* u0[i,j] ./ (c[i,j]^3)
            end
        end
        data_single[:,:,ind] = acoustic_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source_s, source_position_s, receiver_position; pml_len=pml_len, pml_coef=pml_coef);
    end
    
    data_single = Array(data_single)
    
    return data_single
end

# normal operator for Gauss Newton method: (QDF[u])^* QDF[u] u_0 = u_1
# input: u, y, u_0
# output: u_1

function normal_op(wavefield_forward, u0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    
    c = reshape(c, Nx, Ny)
    u0 = reshape(u0, Nx, Ny)
    source_num = size(wavefield_forward,4)
    receiver_num = size(receiver_position,1)
    u1 = SharedArray{Float64}(Nx, Ny, source_num)
    
    data_single = single_scattering(wavefield_forward, u0, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    data_single = SharedArray(data_single)
    
    @inbounds @sync @distributed for ind = 1:source_num
        
        # adjoint of single scattering
        adj_source = data_single[end:-1:1,:,ind];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        utt = zeros(Nx, Ny, Nt)
        @views @. utt[:,:,2:end-1] = (wavefield_forward[:,:,3:end,ind]-2*wavefield_forward[:,:,2:end-1,ind]+wavefield_forward[:,:,1:end-2,ind]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3) * dt
        u1[:,:,ind] = grad0
        
    end
    
    u1 = Array(u1)
    u1 = sum(u1, dims=3)
    u1 = u1[:,:,1]
    u1 = reshape(u1, Nx*Ny,1)
    
    return u1
end


