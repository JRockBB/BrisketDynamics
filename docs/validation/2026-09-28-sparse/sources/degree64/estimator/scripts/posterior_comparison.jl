# Distribution-agreement diagnostics with independent systems as the replication units.
using LinearAlgebra, Statistics

function comparison_cdf(p,i,q)
    R=p.diagnostics.replicates;N=length(p.weights)÷R
    values=[R*sum(p.weights[k]*(p.draws[i,k]<=q) for k in (r-1)*N+1:r*N) for r in 1:R]
    mu=mean(values)
    variance=max(var(values)/R,sum(abs2,p.weights.*((p.draws[i,:].<=q).-mu)),eps(Float64))
    return mu,variance
end

function comparison_second_moments(p,center,scale)
    X=(p.draws.-center)./scale;d,n=size(X);R=p.diagnostics.replicates;N=n÷R
    mu=(X.*p.weights')*X';replicates=zeros(d,d,R)
    for r in 1:R
        ix=(r-1)*N+1:r*N
        replicates[:,:,r]=(X[:,ix].*(R.*p.weights[ix])')*X[:,ix]'
    end
    variance=dropdims(sum((replicates.-mu).^2;dims=3);dims=3)./(R*(R-1))
    for i in 1:d,j in 1:i
        floor=sum(abs2,p.weights.*(X[i,:].*X[j,:].-mu[i,j]))
        variance[i,j]=variance[j,i]=max(variance[i,j],floor,eps(Float64))
    end
    return mu,variance
end

function compare_sampled_posteriors(a,b;quantiles=posterior_quantile)
    # Six estimated MC standard errors are a diagnostic threshold, not a calibration test.
    se=sqrt.(a.diagnostics.mean_mcse.^2+b.diagnostics.mean_mcse.^2)
    standardized=abs.(a.mean-b.mean)./se
    scaled=abs.(a.mean-b.mean)./sqrt.((diag(a.covariance)+diag(b.covariance))./2)
    qa=quantiles(a);qb=quantiles(b);cdf_z=Float64[];cdf_difference=Float64[]
    for i in axes(qa,1),j in axes(qa,2)
        q=(qa[i,j]+qb[i,j])/2;ma,va=comparison_cdf(a,i,q);mb,vb=comparison_cdf(b,i,q)
        push!(cdf_z,abs(ma-mb)/sqrt(va+vb));push!(cdf_difference,abs(ma-mb))
    end
    center=(a.mean+b.mean)./2;scale=sqrt.((diag(a.covariance)+diag(b.covariance))./2)
    ma,va=comparison_second_moments(a,center,scale);mb,vb=comparison_second_moments(b,center,scale)
    moment_z=abs.(ma-mb)./sqrt.(va+vb)
    return (max_mean_mc_z=maximum(standardized),max_mean_posterior_sd=maximum(scaled),
        max_cdf_mc_z=maximum(cdf_z),max_cdf_difference=maximum(cdf_difference),
        max_joint_moment_mc_z=maximum(moment_z),max_joint_moment_difference=maximum(abs.(ma-mb)),
        passed=maximum(standardized)<=6 && maximum(cdf_z)<=6 && maximum(moment_z)<=6)
end

