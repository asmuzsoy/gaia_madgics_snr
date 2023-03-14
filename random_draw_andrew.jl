import Pkg
Pkg.activate("/uufs/chpc.utah.edu/common/home/u6039752/scratch/julia_env/dibs/")


#using Distributed
#addprocs(8)


using Distributed, SlurmClusterManager
addprocs(SlurmManager())

@everywhere println("hello from $(myid()):$(gethostname())")
flush(stdout)

@everywhere begin
    import Pkg
    Pkg.activate("/uufs/chpc.utah.edu/common/home/u6039752/scratch/julia_env/dibs/")
end

@everywhere begin
    #File Handling
    using FITSIO, Serialization, HDF5
    
    #Random Sampling
    using Random, Distributions

    # Stats
    using StatsBase, LinearAlgebra, ProgressMeter
    BLAS.set_num_threads(1)
end


@everywhere begin
    save_starres = true
    dust_on = false 
    dust_iter = 2

    basework = "/uufs/chpc.utah.edu/common/home/u6039752/scratch/working/"
    #out_dir = basework*"2023_02_06/out/"
    out_dir = "../test_new_cov_all/"

    # grvs = h5open("../2023_02_06/test_good.h5");
    # grvs = h5open("test_gaia_data1.h5")
    grvs = h5open(basework*"2022_11_28/sources/gaia_rvs.h5");


#     Cstar = h5read("../2022_11_28/priors/RVS_stellar_zeroweighted_kry_50_95_const.h5","Cstar")
#     Cstarinv = h5read("../2022_11_28/priors/RVS_stellar_zeroweighted_kry_50_95_const.h5","Cstarinv")
#     Vmat_star = h5read("../2022_11_28/priors/RVS_stellar_zeroweighted_kry_50_95_const.h5","Vmat")

#     Vmat_dust = h5read("../2022_11_28/priors/precomp_dust_2_analyticDeriv.h5","Vmat")
#     covdet = h5read("../2022_11_28/priors/precomp_dust_2_analyticDeriv.h5","covdet");
    
    # prior_file = basework*"/2022_11_28/priors/RVS_stellar_zeroweighted_kry_50_95_const.h5"
    prior_file = "modified_prior.h5"
    Cstar = h5read(prior_file,"Cstar") 
    Cstarinv = h5read(prior_file,"Cstarinv") 
    Vmat_star = h5read(prior_file,"Vmat")
    
    Vmat_dust = h5read(basework*"/2022_11_28/priors/precomp_dust_2_analyticDeriv.h5","Vmat")
    covdet = h5read(basework*"/2022_11_28/priors/precomp_dust_2_analyticDeriv.h5","covdet");

    # star_msk, nansum = deserialize("../2022_11_28/priors/star_mask.jdat");

    star_msk, nansum = deserialize(basework*"/2022_11_28/priors/star_mask.jdat");
    msk_inter = deserialize("msk_inter.jdat");
    
    xmin0 = 8460
    xmax0 = 8700
    wavestep = 0.1
    wavex = xmin0:wavestep:xmax0

    nfeat = count(star_msk)
    dind = 1:(nfeat+1):(nfeat^2)
    nvec_star = 50
    dind_star = 1:(nvec_star+1):(nvec_star^2)
    nvec_dust = 2
    dind_dust = 1:(nvec_dust+1):(nvec_dust^2)
    srng = -100:0.1:100
    maxsrng_indx = length(srng)
    sstep=0.01
    sigrng = 0.4:0.01:4
    maxsigrng_indx = length(sigrng)
    sigindx0 = findfirst(sigrng.==1.8)
    sigstep = 0.01
    sigslice = 40
    locGridx = -10:10:10
    hessindx = [1,2,3]
    locGridy = -50:10:50
    hessindy = [5,6,7]
    meanCont = 1.0 #0.95
    diagRenorm = 3.6

    #these are intermediates that will get refilled
    #and rewritten over all the time
    l_msk = length(wavex);
    x_d = zeros(l_msk)
    x_d_spec = zeros(l_msk)
    x_d_sky = zeros(l_msk)
    Xd = zeros(nfeat);
    Xd_var = zeros(nfeat);
    new_gauss = zeros(maxsigrng_indx)

    μ_bd = zeros(l_msk)
    μ_cd = zeros(l_msk)
    μ_dd = zeros(l_msk);
    # some inplace savings?
    AinvVstar = zeros(nfeat,nvec_star)
    M_star = zeros(nvec_star,nvec_star);

    AinvVdust = zeros(nfeat,nvec_dust)
    M_dust = zeros(nvec_dust,nvec_dust);

    CurCovg = zeros(nfeat,nfeat)
    FutCovg = zeros(nfeat,nfeat)

    # these are preallocations per batch
    batsz = 40 #1000

    outb0 = zeros(l_msk,batsz)
    outc0 = zeros(l_msk,batsz)

    outb = zeros(l_msk,batsz)
    outc = zeros(l_msk,batsz)
    outd = zeros(l_msk,batsz);

    outlst = zeros(23,batsz)
    chi2srgh = zeros(length(1:10:length(srng)),batsz);
    chi2sigrgh = zeros(length(1:2:length(sigrng)),batsz);
    chi2sfine = zeros(length(-180:180),batsz);
    chi2sigfine = zeros(length(-90:90),batsz);
    locGrid = zeros(length(locGridx),length(locGridy),3,batsz);
end

@everywhere begin
    function sqrt_safe(x)
        if x < 0
            return -sqrt(abs(x))
        else
            return sqrt(x)
        end
    end

    function gauss_mult_deblend!(Xd, Xd_var, Σ_inv, μ_bd, μ_cd, msk)
        Σ_c = Diagonal(Xd_var)

        Σinvx = (Σ_inv*Xd)

        # export
        fill!(μ_bd,NaN)
        fill!(μ_cd,NaN)

        @views μ_cd[msk].= Σ_c*Σinvx;
        @views μ_bd[msk].= Xd.-μ_cd[msk]
        return
    end

    function gauss_mult_deblend_dust!(Xd, Xd_var, Σ_inv, Vdustfinal, μ_bd, μ_cd, μ_dd, msk)
        Σ_c = Diagonal(Xd_var)

        Σinvx = (Σ_inv*Xd)

        # export
        fill!(μ_bd,NaN)
        fill!(μ_cd,NaN)
        fill!(μ_dd,NaN)

        @views μ_cd[msk].= Σ_c*Σinvx;
        @views μ_dd[msk].= (Vdustfinal*(Vdustfinal'*Σinvx))
        @views μ_bd[msk].= Xd.-μ_cd[msk].-μ_dd[msk]
        return
    end

    function gauss_mult_deblend_dust_derr!(Xd, Xd_var, Σ_inv, Vdustfinal, μ_bd, μ_cd, μ_dd, msk, derr)
        Σ_c = Diagonal(Xd_var)

        Σinvx = (Σ_inv*Xd)

        # export
        fill!(μ_bd,NaN)
        fill!(μ_cd,NaN)
        fill!(μ_dd,NaN)
        fill!(derr,NaN)

        @views μ_cd[msk].= Σ_c*Σinvx;
        @views μ_dd[msk].= (Vdustfinal*(Vdustfinal'*Σinvx))
        @views μ_bd[msk].= Xd.-μ_cd[msk].-μ_dd[msk]

        @views derr .= (-((Vdustfinal*(Vdustfinal'*Σ_inv))-I)*Vdustfinal)*Vdustfinal'
        return
    end

    function gauss_mult_deblend_dust_err!(Xd, Xd_var, Σ_inv, Vdustfinal, Vmat_star, μ_bd, μ_cd, μ_dd, msk, ind, berr, cerr, derr)
        Σ_c = Diagonal(Xd_var)

        Σinvx = (Σ_inv*Xd)

        # export
        fill!(μ_bd,NaN)
        fill!(μ_cd,NaN)
        fill!(μ_dd,NaN)
        fill!(berr,NaN)
        fill!(cerr,NaN)
        fill!(derr,NaN)

        @views μ_cd[msk].= Σ_c*Σinvx;
        @views μ_dd[msk].= Σ_dust*Σinvx;
        @views μ_bd[msk].= Xd.-μ_cd[msk].-μ_dd[msk]

        @views berr .= ((-(Vmat_star*(Vmat_star'*Σ_inv)-I))*Vmat_star)*Vmat_star'
        @views cerr .= (-(Σ_c*Σ_inv-I))*Σ_c
        @views derr .= (-((Vdustfinal*(Vdustfinal'*Σ_inv))-I)*Vdustfinal)*Vdustfinal'
        return
    end
    
    function fast_coverr_sum(Cinv,V)
        xtV = sum(V,dims=1)
        xtVVt = xtV*V'
        return ((xtV*xtV')-((xtVVt*Cinv)*xtVVt'))[1]
    end
    
    function gauss_mult_deblend_dust_sumderr!(Xd, Xd_var, Σ_inv, Vdustfinal, μ_bd, μ_cd, μ_dd, msk)
        Σ_c = Diagonal(Xd_var)

        Σinvx = (Σ_inv*Xd)

        # export
        fill!(μ_bd,NaN)
        fill!(μ_cd,NaN)
        fill!(μ_dd,NaN)

        @views μ_cd[msk].= Σ_c*Σinvx;
        @views μ_dd[msk].= (Vdustfinal*(Vdustfinal'*Σinvx))
        @views μ_bd[msk].= Xd.-μ_cd[msk].-μ_dd[msk]

        return fast_coverr_sum(Σ_inv,Vdustfinal)
    end

    function woodbury_update_inv!(AinvV,M,Ainv,V,dind,Mout)
        mul!(AinvV,Ainv,V)
        mul!(M,V',AinvV)
        M[dind] .+= 1
        Minv = cholesky!(Symmetric(M))
        out = Minv\AinvV'
        Mout .= Ainv
        mul!(Mout,AinvV,out,-1,1)
        return
    end

    function woodbury_update_inv_tst!(AinvV,M,Ainv,Xd,V,dind)
        mul!(AinvV,Ainv,V)
        XdAinvV = reshape(Xd,1,:)*AinvV
        mul!(M,V',AinvV)
        M[dind] .+= 1
        Minvc = cholesky!(Symmetric(M))
        return -(XdAinvV*(Minvc\XdAinvV'))[1]
    end
    
    function woodbury_update_invAndtst!(AinvV,M,Ainv,Xd,V,dind,Mout)
        mul!(AinvV,Ainv,V)
        XdAinvV = reshape(Xd,1,:)*AinvV
        mul!(M,V',AinvV)
        M[dind] .+= 1
        Minv = cholesky!(Symmetric(M))
        out = Minv\AinvV'
        Mout .= Ainv
        mul!(Mout,AinvV,out,-1,1)
        return -(XdAinvV*(Minv\XdAinvV'))[1]
    end

    function woodbury_update_inv_tst_sub!(AinvV,M,Ainv,Xd,V,dind,slrng)
        mul!(AinvV,view(Ainv,:,slrng),view(V,slrng,:))
        XdAinvV = reshape(Xd,1,:)*AinvV
        mul!(M,V',AinvV)
        M[dind] .+= 1
        Minvc = cholesky!(Symmetric(M))
        return -(XdAinvV*(Minvc\XdAinvV'))[1]
    end

    function hessian(locgrid, pts_x, pts_y;dx=0.1,dy=0.05)
        lx, mx, hx = pts_x
        ly, my, hy = pts_y
        Hxx = (locgrid[hx,my]-2*locgrid[mx,my]+locgrid[lx,my])/(dx^2)
        Hyy = (locgrid[mx,hy]-2*locgrid[mx,my]+locgrid[mx,ly])/(dy^2)
        Hxy = (locgrid[hx,hy]-locgrid[hx,ly]-locgrid[lx,hy]+locgrid[lx,ly])/(4*dx*dy)
        det0 = Hxx*Hyy-Hxy^2
        errx= Hyy/det0
        erry= Hxx/det0
        errxy= -Hxy/det0
        #careful, these are variances, you should probably take a sqrt
        return [errx, erry, errxy, 1/Hxx, 1/Hyy]
    end
    
    function gaussian_post(x,x0,s0)
        return 1/sqrt(2π)/s0*exp(-0.5*((x-x0)/s0)^2)
    end

end

@everywhere begin

    function readCorrect(indsubset)
        fill!(outb0,0)
        fill!(outc0,0)
        fill!(outb,0)
        fill!(outc,0)
        fill!(outd,0)
        fill!(outlst,0)
        fill!(chi2srgh,0)
        fill!(chi2sigrgh,0)
        fill!(chi2sfine,0)
        fill!(chi2sigfine,0)
        fill!(locGrid,NaN)

        for (ind,tstind) in enumerate(indsubset)
            x_d_flux = (grvs["flux"])[:,tstind].-meanCont;
            x_d_dflux = grvs["dflux"][:,tstind];
            Xd .= x_d_flux[star_msk];
            
            # Xd .= x_d_flux;

            
            # this diagRenorm is fixed exactly by the sum(banded covariance) due
            # to Gaia RVS spectra correlations induced my some sampling kernel
            Xd_var .= diagRenorm .*x_d_dflux[star_msk].^2
            # Xd_var .= diagRenorm .*x_d_dflux.^2


            Ainv = Diagonal(1 ./Xd_var);
            # Ainv = inv(cholesky(Diagonal(Xd_var) .+ 0.0025));

            CurCov = CurCovg
            FutCov = FutCovg

            woodbury_update_inv!(
                AinvVstar,M_star,
                Ainv,
                Vmat_star,
                dind_star,
                CurCov
            )

            gauss_mult_deblend!(Xd, Xd_var, CurCov, μ_bd, μ_cd, star_msk)

            outlst[1,ind] = μ_bd[star_msk]'*Cstarinv*μ_bd[star_msk]
            outlst[2,ind] = μ_cd[star_msk]'*Ainv*μ_cd[star_msk]

            if save_starres
                outb0[:,ind] .= μ_bd.+meanCont
                outc0[:,ind] .= μ_cd
            end

            if dust_on
                for iter=1:dust_iter
                    Dscale = Diagonal(μ_bd[star_msk].+meanCont)

                    if iter != 1
                        chi2srgh[:,ind] .= 0
                        chi2sigrgh[:,ind] .= 0
                        chi2sfine[:,ind] .= 0
                        chi2sigfine[:,ind] .= 0
                    end

                    ## Rough Loop
                    # dust scan RV rough
                    for (sindxr,svalr) in enumerate(1:10:length(srng))
                        sval = srng[svalr]
                        rmsind = Int(mod1(10 *sval +6,10))
                        whsind = Int(fld1(10 *sval +6,10)-1)
                        wscan = ceil(Int,sigslice*sigrng[sigindx0])
                        slrng = maximum([(1541-whsind-wscan),1]):minimum([(1541-whsind+wscan),nfeat])
                        chi2srgh[sindxr,ind] = woodbury_update_inv_tst_sub!(
                            AinvVdust,M_dust,
                            CurCov,
                            Xd,
                            Dscale*circshift(view(Vmat_dust,:,:,sigindx0,rmsind),(-whsind,0)),
                            dind_dust,
                            slrng
                        ) .+ covdet[sigindx0,rmsind]
                    end

                    minval, minind = findmin(chi2srgh[:,ind])
                    outlst[3,ind] = minval
                    outlst[4,ind] = (1:10:length(srng))[minind]

                    sindx = Int(outlst[4,ind])
                    sval = srng[sindx]

                    # dust scan sigma rough
                    rmsind = Int(mod1(10 *sval +6,10))
                    whsind = Int(fld1(10 *sval +6,10)-1)
                    for (sigindxr,sigvalr) in enumerate(1:2:length(sigrng))
                        sigval = sigrng[sigvalr]
                        wscan = ceil(Int,sigslice*sigval)
                        slrng = maximum([(1541-whsind-wscan),1]):minimum([(1541-whsind+wscan),nfeat])
                        chi2sigrgh[sigindxr,ind] = woodbury_update_inv_tst_sub!(
                            AinvVdust,M_dust,
                            CurCov,
                            Xd,
                            Dscale*circshift(view(Vmat_dust,:,:,sigvalr,rmsind),(-whsind,0)),
                            dind_dust,
                            slrng
                        ) .+ covdet[sigvalr,rmsind]
                    end

                    minval, minind = findmin(chi2sigrgh[:,ind])
                    outlst[5,ind] = minval
                    outlst[6,ind] = (1:2:length(sigrng))[minind]

                    sigindx = Int(outlst[6,ind])
                    sigval = sigrng[sigindx]

                    ## Fine Loop
                    # dust scan RV fine
                    wscan = ceil(Int,sigslice*sigval)
                    for (sindxf,svalf) in enumerate(-180:180)
                        if (0 < sindx+svalf <=maxsrng_indx)
                            sval = srng[sindx+svalf]
                            rmsind = Int(mod1(10 *sval +6,10))
                            whsind = Int(fld1(10 *sval +6,10)-1)
                            slrng = maximum([(1541-whsind-wscan),1]):minimum([(1541-whsind+wscan),nfeat])
                            chi2sfine[sindxf,ind] = woodbury_update_inv_tst_sub!(
                                AinvVdust,M_dust,
                                CurCov,
                                Xd,
                                Dscale*circshift(view(Vmat_dust,:,:,sigindx,rmsind),(-whsind,0)),
                                dind_dust,
                                slrng
                            ) .+ covdet[sigindx,rmsind]
                        end
                    end

                    minval, minind = findmin(chi2sfine[:,ind])
                    outlst[7,ind] = minval
                    outlst[8,ind] = sindx+(-180:180)[minind]

                    sindx = Int(outlst[8,ind])
                    sval = srng[sindx]

                    # dust scan sigma fine
                    rmsind = Int(mod1(10 *sval +6,10))
                    whsind = Int(fld1(10 *sval +6,10)-1)
                    for (sigindxf,sigvalf) in enumerate(-90:90)
                        sigindxt = sigindx+sigvalf
                        if (0 < sigindxt <=maxsigrng_indx)
                            sigval = sigrng[sigindxt]
                            wscan = ceil(Int,sigslice*sigval)
                            slrng = maximum([(1541-whsind-wscan),1]):minimum([(1541-whsind+wscan),nfeat])
                            chi2sigfine[sigindxf,ind] = woodbury_update_inv_tst_sub!(
                                AinvVdust,M_dust,
                                CurCov,
                                Xd,
                                Dscale*circshift(view(Vmat_dust,:,:,sigindxt,rmsind),(-whsind,0)),
                                dind_dust,
                                slrng,
                            ) .+ covdet[sigindxt,rmsind]
                        end
                    end

                    minval, minind = findmin(chi2sigfine[:,ind])
                    outlst[9,ind] = minval
                    outlst[10,ind] = sigindx+(-90:90)[minind]

                    sigindx = Int(outlst[10,ind])
                    sigval = sigrng[sigindx]
                    if iter != dust_iter #just updating the stellar model, don't mind me
                        woodbury_update_inv!(
                            AinvVdust,M_dust,
                            CurCov,
                            Dscale*circshift(view(Vmat_dust,:,:,sigindx,rmsind),(-whsind,0)),
                            dind_dust,
                            FutCov,
                        )
                        
                        gauss_mult_deblend_dust!(
                            Xd,
                            Xd_var,
                            FutCov,
                            Dscale*circshift(view(Vmat_dust,:,:,sigindx,rmsind),(-whsind,0)),
                            μ_bd,
                            μ_cd,
                            μ_dd,
                            star_msk
                            )
                    else # termination step
                        # worth asking hess[2,2]-outlst[9] to quantify error
                        # made in the 40 sigmaslice cut, so center isn't totally repetative
                        for (indoffx,offx) in enumerate(locGridx)
                            for (indoffy,offy) in enumerate(locGridy)
                                cond = ((sigindx+offy)>0)
                                cond &= ((sindx+offx)>0)
                                cond &= ((sigindx+offy)<=maxsigrng_indx)
                                cond &= ((sindx+offx)<=maxsrng_indx)
                                if cond
                                    svalt = srng[sindx+offx]
                                    rmsindt = Int(mod1(10 *svalt +6,10))
                                    whsindt = Int(fld1(10 *svalt +6,10)-1)

                                    if (offx == 0)
                                        locGrid[indoffx,indoffy,1,ind] = woodbury_update_invAndtst!(
                                            AinvVdust,M_dust,
                                            CurCov,
                                            Xd,
                                            Dscale*circshift(view(Vmat_dust,:,:,sigindx+offy,rmsindt),(-whsindt,0)),
                                            dind_dust,
                                            FutCov
                                        ) .+ covdet[sigindx+offy,rmsindt]

                                        sumderr = gauss_mult_deblend_dust_sumderr!(
                                            Xd,
                                            Xd_var,
                                            FutCov,
                                            Dscale*circshift(view(Vmat_dust,:,:,sigindx+offy,rmsindt),(-whsindt,0)),
                                            μ_bd,
                                            μ_cd,
                                            μ_dd,
                                            star_msk,
                                        )

                                        locGrid[indoffx,indoffy,2,ind] = sum(μ_dd[star_msk]./(μ_bd[star_msk].+meanCont))*wavestep
                                        # a more correct form of the calculation propagates the error from berr as well
                                        locGrid[indoffx,indoffy,3,ind] = sqrt_safe(sumderr)*wavestep

                                        if ((offx==0) & (offy==0))
                                            outlst[11,ind] = srng[sindx]
                                            outlst[12,ind] = sigrng[sigindx]

                                            outlst[13,ind] = (μ_bd[star_msk]'*Cstarinv*μ_bd[star_msk])
                                            outlst[14,ind] = μ_cd[star_msk]'*Ainv*μ_cd[star_msk]
                                            outlst[15,ind] = Xd'*FutCov*Xd

                                            outb[:,ind] .= μ_bd.+meanCont
                                            outc[:,ind] .= μ_cd
                                            outd[:,ind] .= μ_dd
                                        end
                                    elseif (hessindy[1] <= indoffy <= hessindy[3])
                                        locGrid[indoffx,indoffy,1,ind] = woodbury_update_inv_tst!(
                                            AinvVdust,M_dust,
                                            CurCov,
                                            Xd,
                                            Dscale*circshift(view(Vmat_dust,:,:,sigindx+offy,rmsindt),(-whsindt,0)),
                                            dind_dust,
                                        ) .+ covdet[sigindx+offy,rmsindt]
                                    end

                                end
                            end
                        end
                        
                        # hessian based error bars
                        varx, vary, varxy, varxx, varyy = hessian(locGrid[:,:,1,ind],hessindx,hessindy;dx=10*sstep,dy=10*sigstep)
                        outlst[16:18,ind] .= varx, vary, varxy
                        # implement fall back for rare negative cases
                        if (outlst[16,ind] > 0) .& (outlst[17,ind] > 0)
                            outlst[19:20,ind] .= sqrt.(outlst[16:17,ind].* 2)
                        elseif (varxx > 0) .& (varyy > 0)
                            outlst[19,ind] = sqrt(2*varxx)
                            outlst[20,ind] = sqrt(2*varyy)
                        else #this only happens we we are close to edge of the grid, toss these cases anyway
                            outlst[19:20,ind] .= NaN
                        end
                        
                        outlst[21,ind] = locGrid[hessindx[2],hessindy[2],2,ind]
                        
                        # extra marginalization step to handle flux error
                        pweight = exp.(0.5*(-locGrid[:,:,1,ind].+locGrid[hessindx[2],hessindy[2],1,ind]))
                        pweight ./= sum(filter(.!isnan,pweight))
                        
                        exval = vcat((locGrid[:,:,2,ind].-3 .*locGrid[:,:,3,ind])[:], (locGrid[:,:,2,ind].+3 .*locGrid[:,:,3,ind])[:])
                        if count(.!isnan.(exval)) > 1
                            minx, maxx = extrema(filter(.!isnan,exval))
                            xrng = range(minx,maxx,length=maxsigrng_indx)

                            fill!(new_gauss,0)
                            for (indoffx,offx) in enumerate(locGridx)
                                for (indoffy,offy) in enumerate(locGridy)
                                    est_flux = locGrid[indoffx,indoffy,2,ind]
                                    sig_flux = locGrid[indoffx,indoffy,3,ind]
                                    if ((.!isnan(est_flux)) & (.!isnan(sig_flux)))
                                        new_gauss .+= gaussian_post.(xrng,Ref(est_flux),Ref(sig_flux))*pweight[indoffx,indoffy]
                                    end
                                end
                            end
                            wvec = ProbabilityWeights(new_gauss)
                            outlst[22,ind] = std(xrng,wvec)
                            outlst[23,ind] = mean(xrng,wvec)
                        else
                            outlst[22,ind] = NaN
                            outlst[23,ind] = NaN
                        end
                    end
                end
            end
        end
        indxs = indsubset[1]
        h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","outlst",outlst)
        if save_starres
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","μ_bd0",outb0)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","μ_cd0",outc0)
        end
        if dust_on
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","μ_bd",outb)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","μ_cd",outc)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","μ_dd",outd)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","chi2srgh",chi2srgh)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","chi2sigrgh",chi2sigrgh)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","chi2sfine",chi2sfine)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","chi2sigfine",chi2sigfine)
            h5write(out_dir*"deblend_scan_"*lpad(indxs,7,"0")*".h5","locGrid",locGrid)
        end
        return
    end
end

itarg = Iterators.partition((1:size(grvs["flux"],2))[msk_inter],batsz);
# itarg = Iterators.partition(1:200,batsz);
println("Batches to Do: ",length(itarg))
println(itarg)
flush(stdout)

@showprogress pmap(readCorrect,itarg)
