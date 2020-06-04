function div_op(f)
    Nx = size(f,1)
    Ny = size(f,2)
    
    div_f = zeros(Nx, Ny)
    
    div_f[2:end-1,:] .+= f[2:end-1,:,1] - f[1:end-2,:,1]
    div_f[1,:] .+= f[1,:,1]
    div_f[end,:] .+= -f[end-1,:,1]
    
    div_f[:,2:end-1] .+= f[:,2:end-1,2] - f[:,1:end-2,2]
    div_f[:,1] .+= f[:,1,2]
    div_f[:,end] .+= -f[:,end-1,2]
    
    return div_f
end

function nabla_op(f)
    Nx = size(f,1)
    Ny = size(f,2)
    
    nabla_f = zeros(Nx, Ny, 2)
    nabla_f[1:end-1,:,1] = f[2:end,:] - f[1:end-1,:]
    nabla_f[:,1:end-1,2] = f[:,2:end] - f[:,1:end-1]
    
    return nabla_f
end

function abs_op(f)
    Nx = size(f,1)
    Ny = size(f,2)
    abs_f = zeros(Nx,Ny)
    abs_f = sqrt.(f[:,:,1].^2 + f[:,:,2].^2)
    return abs_f
end

function TV_projection(g, lambda; iterMax=1000, tau=1/8)
    Nx = size(g,1)
    Ny = size(g,2)
    p0 = zeros(Nx,Ny,2)
    p1 = zeros(Nx,Ny,2)
    
    # main iteration
    for iter = 1:iterMax
        temp = div_op(p0) - g/lambda
        nabla_temp = nabla_op(temp)
        abs_nabla_temp = abs_op(nabla_temp)
        p1 = (p0 + tau*nabla_temp) ./ (1 .+ tau*abs_nabla_temp)
        p0 = copy(p1)
    end
    
    pi_g = div_op(p1);
    u = g - lambda * pi_g;
    return u
end

function TV_projection_dual(g, lambda; iterMax=1000, tau=1/8)
    Nx = size(g,1)
    Ny = size(g,2)
    p0 = zeros(Nx,Ny,2)
    p1 = zeros(Nx,Ny,2)
    
    # main iteration
    for iter = 1:iterMax
        temp = div_op(p0) - g/lambda
        nabla_temp = nabla_op(temp)
        abs_nabla_temp = abs_op(nabla_temp)
        p1 = (p0 + tau*nabla_temp) ./ (1 .+ tau*abs_nabla_temp)
        p0 = copy(p1)
    end
    
    pi_g = div_op(p1);
    return pi_g
end