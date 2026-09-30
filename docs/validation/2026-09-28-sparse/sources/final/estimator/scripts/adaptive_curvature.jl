# Optional observed-curvature oracle for adaptive_conditional.jl. Requires the
# independent curvature_audit.jl functions; no production tolerance/noise change.

"Finish the mode solve using exact innovation scores, avoiding process-residual cancellation."
function polish_conditional_innovations(original;solver_factor=.0001,score_tol=1e-5,max_steps=30)
    ref=curvature_accuracy_ref(original,solver_factor)
    for f in ref.dynamics;f.relin_tol=0.;f.relin_tol_state=0.;end
    u=copy(ref.mean);history=NamedTuple[]
    n=size(ref.root,2);B=Matrix{Float64}(I,n,n)
    for iteration in 0:max_steps
        current=innovation_target(ref,u;gradient=true)
        score=ref.root'*current.gradient;score_norm=norm(score)
        push!(history,(iteration=iteration,objective=current.value,score=score_norm))
        if score_norm<=score_tol
            ref.mean.=u
            return (ref=ref,value=current.value,score=score_norm,history=history)
        end
        iteration<max_steps || error("innovation mode polishing exceeded its iteration budget")
        # Solve the exact score equation with a residual-norm merit function.
        # Near a mode, ODE objective differences can be below numerical resolution
        # even when the score still resolves the direction. Backtracking on the
        # score avoids certifying a mode from a rounded/flat objective. The GN
        # root is a preconditioner, not a covariance or positive-Hessian claim.
        accepted=false
        for attempt in 1:2
            attempt==2 && (B=Matrix{Float64}(I,n,n))
            direction=-B*score;alpha=1.
            for _ in 1:16
                trial=u+alpha*ref.root*direction;next=innovation_target(ref,trial;gradient=true)
                next_score=ref.root'*next.gradient
                if isfinite(next.value) && sum(abs2,next_score)<=(1-1e-4alpha)*score_norm^2
                    # Inverse BFGS is used ONLY to accelerate the score solve.
                    # Returned covariance is independently recomputed from GN
                    # or the full observed Hessian, never from this matrix.
                    s=alpha*direction;y=next_score-score;ys=dot(y,s)
                    if ys>sqrt(eps(Float64))*norm(s)*norm(y)
                        By=B*y
                        B+=((ys+dot(y,By))/ys^2)*(s*s')-(By*s'+s*By')/ys
                        B=Matrix(Symmetric(B))
                    end
                    u=trial;accepted=true;break
                end
                alpha/=2
            end
            accepted && break
        end
        accepted || error("innovation mode polishing could not reduce the score residual (score=$score_norm)")
    end
    error("unreachable innovation-polishing exit")
end

"Polished mode with the GN metric recomputed there; useful for isolating quadrature error."
function polished_gn_conditional_curvature(node,original;kwargs...)
    fit=polish_conditional_innovations(original;kwargs...)
    G=innovation_gn_metric(fit.ref,fit.ref.mean);U=cholesky(Symmetric(G)).U
    return (mode_converged=true,decoder=fit.ref,mean=copy(fit.ref.mean),state_mean=E._conditional_decode(fit.ref,fit.ref.mean),
        root=original.root/U,log_integral=node.log_integral+E.objective(node.graph)-fit.value-sum(log,diag(U);init=0.),
        diagnostics=(method=:polished_gauss_newton,score=fit.score,iterations=length(fit.history)-1,
            process_noise_changed=false))
end

"Correct covariance AND conditional evidence, rejecting unresolved/indefinite curvature."
function observed_conditional_curvature(node,original;steps=(.02,.01),solver_factor=.0001,
                                         curvature_tol=.001,stationarity_tol=.01,polish=false)
    if size(original.root,2)==0
        return (mode_converged=true,decoder=original,mean=copy(original.mean),state_mean=copy(node.mean),root=copy(original.root),
            log_integral=node.log_integral,diagnostics=(method=:exact_no_remainder,stationarity=0.,
                process_noise_changed=false,eigenvalues_clipped=false))
    end
    fit=polish ? polish_conditional_innovations(original;solver_factor) : nothing
    ref=polish ? fit.ref : curvature_accuracy_ref(original,solver_factor)
    # The key is unused for a full basis; pass any present key to the audit API.
    key=first(ref.graph.keys)
    audit=innovation_curvature(ref;key,basis=:full,steps)
    check=last(audit.checks);H=audit.hessian
    length(steps)>=2 && check.change<=curvature_tol && check.asymmetry<=curvature_tol ||
        error("observed conditional curvature is numerically unresolved")
    isposdef(Symmetric(H)) || error("observed conditional Hessian is not positive definite")
    U=cholesky(Symmetric(H)).U
    stationarity=norm(U'\audit.gradient)
    stationarity<=stationarity_tol || error("conditional mode is not stationary in observed-curvature units")
    logvolume=-sum(log,diag(U))
    objective_correction=audit.objective-E.objective(node.graph)
    return (mode_converged=polish,decoder=ref,mean=copy(ref.mean),
        state_mean=E._conditional_decode(ref,ref.mean),root=original.root/U,
        log_integral=node.log_integral+logvolume-objective_correction,
        diagnostics=(method=:observed_full,step_change=check.change,asymmetry=check.asymmetry,
            stationarity=stationarity,log_volume_correction=logvolume,
            objective_correction=objective_correction,solver_factor=solver_factor,
            process_noise_changed=false,eigenvalues_clipped=false,polished=polish,
            polish_steps=polish ? length(fit.history)-1 : 0))
end
