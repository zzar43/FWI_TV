function eval_tv(f)
    Nx = size(f,1)
    Ny = size(f,2)
    
    tv_f = 0
    
    for i = 1:Nx-1
        for j = 1:Ny-1
            tv_f += sqrt((f[i+1,j]-f[i,j])^2 + (f[i,j+1]-f[i,j])^2)
        end
    end
    return tv_f
end


function obj_func_l2(received_data, x, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=50)
    
    c = reshape(x, Nx, Ny)
    
    data = multi_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    
    return 0.5*norm(received_data-data,2)^2
end


function obj_func_tv(received_data, lambda, x, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=50)
    
    c = reshape(x, Nx, Ny)
    
    data = multi_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    
    return 0.5*norm(received_data-data,2)^2 + lambda * eval_tv(c)
end

function grad_l2(received_data, x, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=50)
    
    c = reshape(x, Nx, Ny)
    
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        @views adj_source = data_forward - received_data[:,:,ind]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3)
        grad[:,:,ind] = grad0
        
        # evaluate the objctive function
        obj_value[ind] = 0.5*norm(data_forward-received_data[:,:,ind],2).^2
    end
    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]

    obj_value = sum(obj_value)
    grad = reshape(grad, Nx*Ny, 1)
    return obj_value, grad
end

function grad_tv(received_data, lambda, x, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=20, pml_coef=50, tv_iter_time=1000)
    
    c = reshape(x, Nx, Ny)
    
    source_num = size(source_position,1)
    grad = SharedArray{Float64}(Nx, Ny, source_num)
    obj_value = SharedArray{Float64}(source_num)
    
    @inbounds @sync @distributed for ind = 1:source_num
        u, data_forward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position[ind,:]', receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        @views adj_source = data_forward - received_data[:,:,ind]
        @views adj_source = adj_source[end:-1:1,:];
        v, data_backward = acoustic_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, adj_source, receiver_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
        
        utt = similar(u)
        @views @. utt[:,:,2:end-1] = (u[:,:,3:end]-2u[:,:,2:end-1]+u[:,:,1:end-2]) ./ dt^2;
        @views @. utt = 2 ./ c^3 .* utt .* v[:,:,end:-1:1];
        grad0 = sum(utt,dims=3)
        grad[:,:,ind] = grad0
        
        # evaluate the objctive function
        obj_value[ind] = 0.5*norm(data_forward-received_data[:,:,ind],2).^2
    end
    grad = Array(grad)
    grad = sum(grad, dims=3)
    grad = grad[:,:,1]

    obj_value = sum(obj_value) + lambda * eval_tv(c)
    grad = reshape(grad, Nx, Ny)
    
    grad_x = TV_projection(grad, lambda; iterMax=tv_iter_time);
    grad_x = reshape(grad_x, Nx*Ny, 1)
    return obj_value, grad_x
end