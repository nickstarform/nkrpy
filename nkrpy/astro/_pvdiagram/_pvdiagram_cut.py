
        if config['mcmc']:
            nll = lambda *args: -lnlike(*args)
            initial = np.array([mass_est, 0, np.log(f_true)])
            initial += initial * np.random.randn(initial.shape[0])
            ndim, nwalkers = len(initial), 50
            embed() if config['interactive'] else None
            result = optimize.minimize(nll, initial, args=(x, y, yerr))
            pos = result["x"] + result["x"] * np.random.randn(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
            pos, prob, state = sampler.run_mcmc(pos, 800)
            pax.cla()
            for j in range(ndim):
                for k in range(nwalkers):
                    pax.plot(sampler.chain[k,:,j])

                pfig.savefig(f"{config['savepath']}/burned_steps_{j}.pdf")

                pax.cla()
            lfig.clf()
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, 1000, rstate0=state)
            osamples = sampler.chain
            samples = osamples.copy().reshape((-1, ndim))
            af = sampler.acceptance_fraction
            print("Mean acceptance fraction:", np.mean(af))
            truths = np.median(samples, axis=0)
            print('Plotting Emcee Fits')

            m_true = np.median(samples, axis=0)[0]
            config['EMCEE-SAMPLES'] = [samples, prob]

            pax.cla()
            pax.scatter(x, y, color='red',lw=10,marker='.')
            xdl = np.linspace(-10,-0.25) 
            xdr = np.linspace(0.25, 10)
            for m, v, lnf in samples[np.random.randint(samples.shape[0], size=100)]:
                pax.plot(xdl, fit(xdl, m), color="k", alpha=0.3)
                pax.plot(xdr, fit(xdr, m), color="k", alpha=0.3)  
            pax.plot(xdl, fit(xdl, m_true),color='cyan')
            pax.plot(xdr, fit(xdr, m_true),color='cyan')
            pax.set_xlim(-3, 3)
            pax.set_ylim(0, 4)
            pax.set_aspect(1./pax.get_data_ratio())
            pfig.savefig(f"{config['savepath']}/pvfit.pdf",dpi=300)

            cfig = corner.corner(samples, labels=["$m$", "$v$", "$\ln\,f$"],
                                  truths=truths)
            cfig.savefig(f"{config['savepath']}/triangle.pdf",dpi=600)
            cfig.clf()
            """
            samples[:, 1] = np.exp(samples[:, 1])
            m_mcmc, f_mcmc = map(lambda v: (v[1], v[1]-v[0]),
                                         zip(*np.percentile(samples, [16, 50, 84],
                                                            axis=0)))
            print(f'M:{m_true}..{m_mcmc}\nE:{f_true}..{f_mcmc}')
            ax.plot(xl, inv(xl, m_true ), color="C", alpha=1,label=f'Mass: {m_true:.2f} M$_\odot$')
            ax.plot(xr, inv(xr, -1.* m_true ), color="C", alpha=1)
            eax.plot(xl, inv(xl, m_true ), color="C", alpha=1,label=f'Mass: {m_true:.2f} M$_\odot$')
            eax.plot(xr, inv(xr, -1.* m_true), color="C", alpha=1)
            ebar = plt.colorbar(ecs, ax=eax, fraction=0.046, pad=0.04)
            mass = m_true
            efig.tight_layout()
            eax.set_aspect(1./eax.get_data_ratio())
            efig.savefig(f"{config['savepath']}/pvfit-xtr.pdf",dpi=300)
            """
            pax.cla()
            for j in range(ndim):
                for k in range(nwalkers):
                    pax.plot(osamples[k,:,j])

                pfig.savefig(f"steps_{j}.pdf")