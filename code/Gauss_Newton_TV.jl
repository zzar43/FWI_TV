using Distributed, LinearAlgebra, Printf

# CG iteration for Gauss-Newton
# solve: B_k p_k = - grad_k
# input: u, y, grad, normal_op_handle, iterMax
# output: delta_u1
# function handle:
# normal_op_handle(x) = normal_op(y, x, c, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50);
function CG_iteration(grad, normal_op_handle; tol=0, iterMax=10)
    # println("    Start CG iteration. ||grad f|| = ", norm(grad))
    @printf "    Start CG iteration. ||grad f|| = %1.5e, " norm(grad)
    if tol == 0
        tol = min(0.5, sqrt(norm(grad))) * norm(grad)
        # println("   tol = ", tol)
        @printf "tol = %1.5e\n" tol
    else
        tol = tol * norm(grad)
        # println("   tol = ", tol)
        @printf "tol = %1.5e\n" tol
    end

    p0 = 0 * grad[:]
    p1 = 0 * grad[:]
    r0 = copy(grad)
    r1 = 0 * r0
    d0 = -1 * r0
    d1 = 0 * d0

    # println("   iteration time: ")
    for k = 1:iterMax
        # print(k, ", ")
        theta = normal_op_handle(d0)
        alpha = sum(r0 .* r0) / sum(d0 .* theta)
        p1 = p0 + alpha * d0
        r1 = r0 + alpha * theta
        if norm(r1) < tol
            # println("   ||r1|| = ", norm(r1), " < tol, break.")
            @printf "    iteration: %1.0f, ||r1|| = %1.5e < tol, break\n" k norm(r1)
            break
        else
            # println("   iteration: ", k, ", ||r1|| = ", norm(r1))
            @printf "    iteration: %1.0f, ||r1|| = %1.5e\n" k norm(r1)
        end
        beta = sum(r1 .* r1) / sum(r0 .* r0)
        d1 = -r1 + beta * d0
        
        # update
        p0 = copy(p1)
        r0 = copy(r1)
        d0 = copy(d1)

        if k == iterMax
            print("    iteration time = iterMax, please increase iterMax.")
        end
    end
    println("    CG iteration done.")
    return p1
end

function compute_obj_TV_fn(received_data, c, c0, TV_lambda, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    c = reshape(c, Nx, Ny)
    c0 = reshape(c0, Nx, Ny)
    data_forward = multi_solver_no_wavefield(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    return 0.5 * norm(data_forward-received_data,2)^2+TV_lambda*eval_tv(c-c0)
end

function compute_grad(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
    c = reshape(c, Nx, Ny)

    wavefield_forward, data_forward = multi_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    adjoint_source = data_forward - received_data

    grad = adjoint_op(wavefield_forward, adjoint_source, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    return grad
end

function compute_CG_step(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; CG_tol=0, CG_iterMax=10, pml_len=30, pml_coef=50)

    c = reshape(c, Nx, Ny)

    wavefield_forward, data_forward = multi_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    adjoint_source = data_forward - received_data
    grad = adjoint_op(wavefield_forward, adjoint_source, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    normal_op_handle(x) = normal_op(wavefield_forward, x, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    update_step = CG_iteration(grad, normal_op_handle; tol=CG_tol, iterMax=CG_iterMax)

    return 0.5 * norm(adjoint_source,2)^2, update_step
end

function compute_CG_TV_step(received_data, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; CG_tol=0, CG_iterMax=10, TV_lambda=0, TV_iterMax=1000, pml_len=30, pml_coef=50)

    c = reshape(c, Nx, Ny)

    wavefield_forward, data_forward = multi_solver(c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    adjoint_source = data_forward - received_data
    grad = adjoint_op(wavefield_forward, adjoint_source, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)

    # TV projection
    if TV_lambda == 0
        TV_lambda = 0.01 * norm(grad)
        @printf "    TV projection: ||grad|| = %1.5e, TV_lambda = %1.5e\n" norm(grad) TV_lambda
    else
        @printf "    TV projection: TV_lambda = %1.5e\n" TV_lambda
    end
    grad_tv = TV_projection(reshape(grad,Nx,Ny), TV_lambda; iterMax=TV_iterMax);
    grad_tv = grad_tv[:]

    normal_op_handle(x) = normal_op(wavefield_forward, x, c, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
    update_step = CG_iteration(grad_tv, normal_op_handle; tol=CG_tol, iterMax=CG_iterMax)

    return 0.5 * norm(adjoint_source,2)^2, update_step
end


# ===============
function line_search_backtracking(eval_fn, xk, fk, gradk, alpha, min_value, max_value; rho=0.9, c=0.9, maxSearchTime=30)
    pk = -gradk
    xk = xk[:]
    @printf "Start line search. fk: %1.5e\n" fk
    xkk = update_fn(xk, alpha, gradk, min_value, max_value)
    fk1 = eval_fn(xkk,xk)
    @printf "    alpha: %1.5e" alpha
    @printf "    fk1: %1.5e" fk1
    @printf "    fk-c*alpha*gradk^2: %1.5e\n" (fk + c*alpha*sum(gradk.*pk))
    
    searchTime = 0
    for iter = 1:maxSearchTime
        if fk1 <= (fk + c*alpha*sum(gradk.*pk))
            break
        end
        alpha = rho * alpha
        xkk = update_fn(xk, alpha, gradk, min_value, max_value)
        fk1 = eval_fn(xkk,xk)   
        @printf "    alpha: %1.5e" alpha
        @printf "    fk1: %1.5e" fk1
        @printf "    fk-c*alpha*gradk^2: %1.5e\n" (fk + c*alpha*sum(gradk.*pk))
        searchTime += 1
    end
    
    if fk1 > fk + c*alpha*sum(gradk.*pk)
        println("Line search failed. Search time: ", searchTime, ". Try to decrease search coef c.")
        alpha = 0
    elseif fk1 == NaN
        println("Line search failed. Search time: ", searchTime, ". Function value is NaN.")
    else
        println("Line search succeed. Search time: ", searchTime, ".")
    end

    return alpha
end

function update_fn(xk, alphak, gradk, min_value, max_value)
    xk = xk[:]
    xk1 = xk - alphak * gradk
    if min_value != 0
        xk1[findall(ind->ind<min_value,xk1)] .= min_value
    end
    if max_value != 0
        xk1[findall(ind->ind>max_value,xk1)] .= max_value
    end
    return xk1
end


# function handle
# eval_fn(x, x0) = compute_obj_fn(received_data, x, x0, TV_lambda, rho, Nx, dx, Ny, dy, Nt, dt, source, source_position, receiver_position; pml_len=30, pml_coef=50)
# eval_grad(x) = compute_grad(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; pml_len=pml_len, pml_coef=pml_coef)
# eval_CG_step(x) = compute_CG_step(received_data, x, rho, Nx, h, Ny, h, Nt, dt, source, source_position, receiver_position; CG_tol=0, CG_iterMax=10, pml_len=30, pml_coef=50);

# Gauss Newton method:
function Gauss_Newton(eval_fn, eval_CG_step, x0, iterNum, min_value, max_value; alpha=1, rho=0.9, c=1e-10, maxSearchTime=5, threshold=1e-10)
    println("----------------------------------------------------------------")
    println("Start Gauss Newton method:")
    println("----------------------------------------------------------------")
    xk = convert(Array{Float64,1}, x0[:])
    xk1 = convert(Array{Float64,1}, x0[:])
    fn_value = zeros(iterNum+1)
    
    # this fk is not true
    fk, stepk = eval_CG_step(xk)
    fn_value[1] = fk
    
    for iter = 1:iterNum
        println("----------------------------------------------------------------")
        println("Main iteration: ", iter)
        
        # Line search
        alpha0 = line_search_backtracking(eval_fn, xk, fk, -stepk, alpha, min_value, max_value; rho=rho, c=c, maxSearchTime=maxSearchTime)
        # println(alpha0)
        if alpha0 == 0
            println("----------------------------------------------------------------")
            println("Line search Failed. Try decrease line search coef alpha. Interupt.")
            println("----------------------------------------------------------------")
            break
        else
            xk1 = update_fn(xk, alpha0, -stepk, min_value, max_value)
        end
        
        # Compute gradient for next iteration
        println("----------------------------------------------------------------")
        if iter == iterNum 
            @printf "fk: %1.5e " fk
            println("Iteration is done. \n")
            println("----------------------------------------------------------------\n")
            break;
        end
        # the fk from CG function is not right, should be evaluated by eval_fn again
        fk, stepk = eval_CG_step(xk1)
        fk = eval_fn(xk1, xk)
        xk = copy(xk1)
        fn_value[iter+1] = fk
        if fk <= threshold
            @printf "fk: %1.5e " fk
            println("Iteration is done.")
            println("----------------------------------------------------------------\n")
            break
        end
    end

    return xk, fn_value
end
