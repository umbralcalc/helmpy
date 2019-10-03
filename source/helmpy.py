import numpy as np

# Initialize the 'helmpy' method class
class helmpy:
    

    def __init__(self,
                 helm_type,                          # Must initialise with a disease type declared - types available: 'STH', 'SCH' and 'LF'
                 path_to_helmpy_directory,           # Must initialise with a directory declared
                 suppress_terminal_output=False      # Set this to 'True' to remove terminal messages
                 ):

        self.helm_type = helm_type 
        self.path_to_helmpy_directory = path_to_helmpy_directory 

        self.helmpy_frontpage
        self.add_treatment_prog
        self.run_full_stoch
        self.run_meanfield
        self.worm_STH_stationary_sampler
        self.egg_STH_pulse_sampler
        self.treatment_times = None
        self.treatment_coverages = None
        self.compliance_params = None
        self.migration_mode = False
        self.suppress_terminal_output = suppress_terminal_output
        self.drug_efficacy = 1.0

        # Directory names can be changed here if necessary
        self.chains_directory = 'chains/' 
        self.output_directory = 'data/'
        self.plots_directory = 'plots/'
        self.source_directory = 'source/'     

        if self.helm_type == 'STH':
            # Default is one grouping with the same parameters in cluster '1'
            self.default_parameter_dictionary = {
                                                   'mu':   [1.0/70.0],      # Human death rate (per year)
                                                   'mu1':  [0.5],           # Adult worm death rate (per year)
                                                   'mu2':  [26.0],          # Reservoir (eggs and larvae) death rate (per year)
                                                   'R0':   [2.5],           # Basic reproduction number within grouping
                                                   'k':    [0.3],           # Inverse-clumping factor within grouping
                                                   'gam':  [0.08],          # Density dependent fecundity power-law scaling z = exp(-gam)
                                                   'Np':   [100],           # Number of people within grouping
                                                   'spi':  [1],             # Spatial index number of grouping - modify this only if varying spatially in clusters
                                                   'r+':   [0.0],           # Migration i x i matrix - the migration rate in from each of the i clusters (per year) 
                                                   'r-':   [0.0],           # Migration i x i matrix - the migration rate out to each of the i clusters (per year) 
                                                   'Nm':   [1]              # Migrant number per event (global parameter) - must be integer!!!
                                                 }
            # Default is one grouping with the same initial conditions in cluster '1'
            self.default_initial_conditions = {
                                                 'M':          [2.6],      # Initial mean total worm burden within grouping
                                                 'FOI':        [1.25],     # Initial force of infection (per year) within grouping
                                                 'wormlist':   [],         # Optional initialisation of the separate worm burdens of individuals in each grouping in a list of lists 
                                                 'lamlist':    []          # Optional initialisation of the separate uptake rates of individuals in each grouping in a list of lists 
                                               }

        self.parameter_dictionary = self.default_parameter_dictionary
        self.initial_conditions = self.default_initial_conditions 


    # If new groupings have been added to parameters or initial conditions, fix the dimensions to match in all keys of the dictionary where not specified
    def fix_groupings(self):   
    
        # If the number of people are set in new groupings then create lists of equivalent size with default values for all parameters and initial conditions
        # which are not explictly set from the beginning - this essentially means that the number of people parameter is the most important to set initially
        for key in self.parameter_dictionary:
            if (len(self.parameter_dictionary[key]) != len(self.parameter_dictionary['Np'])) and (key != 'r+') and (key != 'r-') and (key != 'Nm'):

                # Multiply lists by number of new groupings
                self.parameter_dictionary[key] = self.parameter_dictionary[key]*len(self.parameter_dictionary['Np'])
            
        for key in self.initial_conditions:
            if (len(self.initial_conditions[key]) != len(self.parameter_dictionary['Np'])) and (key != 'wormlist') and (key != 'lamlist'):
 
                # Multiply lists by number of new groupings
                self.initial_conditions[key] = self.initial_conditions[key]*len(self.parameter_dictionary['Np'])
  

    def add_treatment_prog(self,
                           treatment_times,              # Input a list of treatment times for all clusters and the code will match to the nearest Poisson timesteps
                           treatment_coverages=None,     # Input a list of lists matching the chosen groupings which give the effective coverage fraction in each age bin and clusters
                           compliance_params=None,       # OR - Input a list of lists twice the length above, giving: alpha = Pr(treated this round | treated last round) and
                                                         #      beta = Pr(treated this round | NOT treated last round) parameter choices for the systematic non-compliance pattern in
                                                         #      the Markovian model - Note also that first round alpha entry is just the initial coverage probability
                           drug_efficacy=1.0             # Optional mean fraction of worms killed when treated - default is perfect efficacy
                           ):

        # Fix the dimensions for all of the groupings
        self.fix_groupings()

        # Define new quantities in the class
        self.treatment_times = np.asarray(treatment_times)

        # Fix the drug efficacy
        self.drug_efficacy = drug_efficacy
        
        # If overall coverage pattern is specified (instead of variable compliance) then define new class quantity
        if treatment_coverages is not None: self.treatment_coverages = np.asarray(treatment_coverages)

        # If random compliance patterns are specified then define new class quantity
        if compliance_params is not None: self.compliance_params = np.asarray(compliance_params)
    

    # Implements a Gillespie algorithm (https://en.wikipedia.org/wiki/Gillespie_algorithm) of the system, running a full stochastic simulation
    # while computing the ensemble mean and ensemble variance as well as the upper and lower limits of the 68 confidence region and outputting to file
    def run_full_stoch(self,
                       runtime,                    # Set the total time of the run in years
                       realisations,               # Set the number of stochastic realisations for the model
                       do_nothing_timescale,       # Set a timescale (in years) short enough such that an individual is expected to stay in the same state
                       output_filename,            # Set a filename for the data to be output in self.output_directory
                       timesteps_snapshot=[],      # Optional - output a snapshot of the worm burdens in each cluster after a specified number of steps in time 
                       mf_migrations=False,        # Optional - use mean field egg count distribution to compute reservoir amplitudes while updating the ensemble mean worm burden
                       mf_migrations_fixed=False   # Optional - set to True if eus migration pulses are drawn from egg count distributions with parameters fixed to the initial conditions
                                                   #          - set to False (default) if eus migration pulses are drawn from egg count distributions with their ensemble means updated 
                       ):

        # Terminal front page when code runs...
        if self.suppress_terminal_output == False: self.helmpy_frontpage()

        # Fix the dimensions for all of the groupings
        self.fix_groupings()

        if self.helm_type == 'STH':
            
            # Set parameter values, initial conditions and cluster references for each realisation
            mus = np.asarray(self.parameter_dictionary['mu'])
            mu1s = np.asarray(self.parameter_dictionary['mu1'])
            mu2s = np.asarray(self.parameter_dictionary['mu2'])
            R0s = np.asarray(self.parameter_dictionary['R0'])
            ks = np.asarray(self.parameter_dictionary['k'])
            gams = np.asarray(self.parameter_dictionary['gam'])
            Nps = np.asarray(self.parameter_dictionary['Np'])
            spis = np.asarray(self.parameter_dictionary['spi'])
            rps = np.asarray(self.parameter_dictionary['r+'])
            rms = np.asarray(self.parameter_dictionary['r-'])
            Ms = np.asarray(self.initial_conditions['M'])
            FOIs = np.asarray(self.initial_conditions['FOI'])

            # Check to see if inter-cluster migration has been specified
            if rps.any() != 0.0 or rms.any() != 0.0: self.migration_mode = True 

            # Find unique cluster references
            uspis = np.unique(spis)

            # Obtain the number of clusters
            numclus = len(uspis)

            lam_ind_perclus = []
            ws_ind_perclus = []
            Ms_ind_perclus = []
            FOIs_ind_perclus = []
            R0s_ind_perclus = []
            mus_ind_perclus = []
            mu1s_ind_perclus = []
            mu2s_ind_perclus = []
            ks_ind_perclus = []
            gams_ind_perclus = []
            Nps_ind_perclus = []

            # Function which maps from worms to eggs in the standard migration model
            def worm_to_egg_func(wormvals,gamvals):
                return (1.0-(2.0**(1.0-wormvals.astype(float))))*wormvals.astype(float)*(np.exp(-gamvals*(wormvals.astype(float)-1.0)))

            # If treatment has been specified, allocate memory
            if self.treatment_times is not None: 
                if self.treatment_coverages is not None: 
                    cov_ind_perclus = []
                if self.compliance_params is not None: 
                    comp_ind_perclus = []
                    last_round_behaviour_ind_perclus = []

            # If migration has been specified, allocate memory for reservoir pulses
            if self.migration_mode == True: eggpulse_ind_perclus = []

            if self.suppress_terminal_output == False: print('Setting initial conditions...')

            # Slow way to initialise a sampled pickup rate 'lambda', initial worm burden, initial worm uptake time and initial worm death time per individual per cluster
            for i in range(0,len(uspis)):
                
                lams_ind_clus = np.empty((0,realisations),float)
                ws_ind_clus = np.empty((0,realisations),float)
                Ms_ind_clus = np.empty((0,realisations),float)
                FOIs_ind_clus = np.empty((0,realisations),float)
                R0s_ind_clus = np.empty((0,realisations),float)
                mus_ind_clus = np.empty((0,realisations),float)
                mu1s_ind_clus = np.empty((0,realisations),float)
                mu2s_ind_clus = np.empty((0,realisations),float)
                ks_ind_clus = np.empty((0,realisations),float)
                gams_ind_clus = np.empty((0,realisations),float)
                Nps_ind_clus = np.empty((0,realisations),float)
  
                # If treatment has been specified, allocate memory
                if self.treatment_times is not None: 
                    if self.treatment_coverages is not None: cov_ind_clus = np.empty((len(self.treatment_times),0,realisations),float)

                    # If non-compliance pattern has been specified, allocate memory
                    if self.compliance_params is not None: 
                        comp_ind_clus = np.empty((2*len(self.treatment_times),0,realisations),float)
                        lr_behaviour_ind_clus = np.empty((0,realisations),float)

                # If migration has been specified, allocate memory for reservoir pulses
                if self.migration_mode == True: eggpulse_ind_clus = np.empty((0,realisations),float)

                # Loop over groupings and stack the arrays
                for j in range(0,len(Nps[spis==uspis[i]])):
                
                    # If list of individual uptake rates has not been specified, draw values from the initial gamma distribution with k
                    if len(self.initial_conditions['lamlist']) == 0:
                        # Draw from lambda ~ Gamma(k,k) for each individual and realisation of pickup rate 
                        # The values in each age bin are also sorted in order to match worm burdens for optimised approach to stationarity
                        lams_ind_clus = np.append(lams_ind_clus,np.sort(np.random.gamma(ks[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)), \
                                                                       (1.0/ks[spis==uspis[i]][j])*np.ones((Nps[spis==uspis[i]][j],realisations)), \
                                                                        size=(Nps[spis==uspis[i]][j],realisations)),axis=0),axis=0)

                    # If list of individual uptake rates has been specified, set the values in each grouping and create the matrix of realisations
                    if len(self.initial_conditions['lamlist']) > 0: 
                        lams_ind_clus = np.append(lams_ind_clus,np.tensordot(np.asarray(self.initial_conditions['lamlist'])[spis==uspis[i]][j],np.ones(realisations),axes=0),axis=0)

                    # If list of individual worm burdens has not been specified, draw values from the initial M
                    if len(self.initial_conditions['wormlist']) == 0:
                        # Draw an individual's worm burden realisations from a negative binomial with initial conditions set
                        # The values in each age bin are also sorted in order to match pickup rates for optimised approach to stationarity
                        ws_ind_clus = np.append(ws_ind_clus,np.sort(np.random.negative_binomial(ks[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)), \
                                                                   ((1.0+(Ms[spis==uspis[i]][j]/ks[spis==uspis[i]][j]))**(-1.0))*np.ones((Nps[spis==uspis[i]][j],realisations)), \
                                                                    size=(Nps[spis==uspis[i]][j],realisations)),axis=0),axis=0)

                    # If list of individual worm burdens has been specified, set the values in each grouping and create the matrix of realisations
                    if len(self.initial_conditions['wormlist']) > 0: 
                        ws_ind_clus = np.append(ws_ind_clus,np.tensordot(np.asarray(self.initial_conditions['wormlist'])[spis==uspis[i]][j],np.ones(realisations),axes=0),axis=0)

                    # Set initial mean worm burden, force of infection, R0, human death rate, worm death rate and eggs/larvae death rate for each individual and realisation
                    Ms_ind_clus = np.append(Ms_ind_clus,Ms[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    FOIs_ind_clus = np.append(FOIs_ind_clus,FOIs[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    R0s_ind_clus = np.append(R0s_ind_clus,R0s[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    mus_ind_clus = np.append(mus_ind_clus,mus[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    mu1s_ind_clus = np.append(mu1s_ind_clus,mu1s[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    mu2s_ind_clus = np.append(mu2s_ind_clus,mu2s[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    ks_ind_clus = np.append(ks_ind_clus,ks[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    gams_ind_clus = np.append(gams_ind_clus,gams[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)
                    Nps_ind_clus = np.append(Nps_ind_clus,Nps[spis==uspis[i]][j]*np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)

                    # If treatment has been specified, give a coverage fraction for each individual to draw from
                    if self.treatment_times is not None: 
                        if self.treatment_coverages is not None: cov_ind_clus = np.append(cov_ind_clus,np.tensordot(self.treatment_coverages[spis==uspis[i]][j], \
                                                                                                       np.ones((Nps[spis==uspis[i]][j],realisations)),axes=0),axis=1)

                        # If non-compliance pattern has been specified, store the conditional probabilities for each individual of the cluster
                        if self.compliance_params is not None: 
                            comp_ind_clus = np.append(comp_ind_clus,np.tensordot(self.compliance_params[spis==uspis[i]][j], \
                                                                                 np.ones((Nps[spis==uspis[i]][j],realisations)),axes=0),axis=1)
                            lr_behaviour_ind_clus = np.append(lr_behaviour_ind_clus,np.ones((Nps[spis==uspis[i]][j],realisations)),axis=0)

                    # If migration has been specified, allocate memory for reservoir pulses
                    if self.migration_mode == True: eggpulse_ind_clus = np.append(eggpulse_ind_clus,np.zeros((Nps[spis==uspis[i]][j],realisations)),axis=0)

                # Append all of the cluster-by-cluster lists
                lam_ind_perclus.append(lams_ind_clus)
                ws_ind_perclus.append(ws_ind_clus)
                Ms_ind_perclus.append(Ms_ind_clus)
                FOIs_ind_perclus.append(FOIs_ind_clus)
                R0s_ind_perclus.append(R0s_ind_clus)
                mus_ind_perclus.append(mus_ind_clus)
                mu1s_ind_perclus.append(mu1s_ind_clus)
                mu2s_ind_perclus.append(mu2s_ind_clus)
                ks_ind_perclus.append(ks_ind_clus)
                gams_ind_perclus.append(gams_ind_clus)
                Nps_ind_perclus.append(Nps_ind_clus)

                # If treatment has been specified, append lists
                if self.treatment_times is not None: 
                    if self.treatment_coverages is not None: 
                        cov_ind_perclus.append(cov_ind_clus)
                    if self.compliance_params is not None: 
                        comp_ind_perclus.append(comp_ind_clus)
                        last_round_behaviour_ind_perclus.append(lr_behaviour_ind_clus)

                # If migration has been specified, allocate memory for reservoir pulses
                if self.migration_mode == True: eggpulse_ind_perclus.append(eggpulse_ind_clus)
      
        if self.suppress_terminal_output == False: 

             print('                       ')
             print('Total number of individuals: ' + str(np.sum(Nps)))
             print('Number of clusters: ' + str(numclus))
             print('                       ')

             if self.migration_mode == True:
                 print('Inter-cluster migration has been enabled...')
                 print('                                           ')

             if self.treatment_times is not None: 
                 print('Treatments are to be performed on the nearest times to: ' + str(self.treatment_times) + ' years')
                 print('                                                                                               ')

             print('Now running full stochastic simulation for ' + str(runtime) + ' years...')
             print('                                                                        ')

        # If treatment has been specified, allocate memory for realisations of the post-last-treatment prevalence per cluster 
        if self.treatment_times is not None: treat_prevs_perclus = []

        # If migration has been specified, allocate memory for the ensemble means in the previous step to use in generating egg pulses
        if self.migration_mode == True: last_ensM = Ms

        count_steps = 0
        output_data = []
        time = 0.0 # Initialise time and loop over drawn timestep
        while time < runtime:

            # Initialise snapshot output lists of the ensemble mean, ensemble variance and 68% upper and lower confidence limits in the mean worm burden per cluster
            ensM_perclus_output = []
            ensV_perclus_output = [] 
            ensup68CL_perclus_output = []
            enslw68CL_perclus_output = []
   
            # If treatment has been specified, initialise snapshot output lists of the ensemble mean and ensemble variance in the mean worm 
            # burden per cluster where the realisations which have been lost to the m(t) = 0 attractor post-treatment have been removed
            if self.treatment_times is not None:
                ensM_zeros_removed_perclus_output = []
                ensV_zeros_removed_perclus_output = []

            # Generate an exponentially-distributed timestep
            timestep = np.random.exponential(scale=do_nothing_timescale)

            # If treatment has been specified, store previous time
            if self.treatment_times is not None: old_time = time

            # Update overall time with timestep
            time += timestep    

            # Count the number of steps performed in time
            count_steps += 1 

            # If treatment has been specified, check to see if this is the time to treat by setting a treatment index 
            if self.treatment_times is not None: treat_ind = (old_time < self.treatment_times)*(self.treatment_times <= time)

            # If migration has been specified, initialise count downward to determine other migrations as well as memory  
            # for the sum of pulses in each cluster and the number of migrants per event
            if self.migration_mode == True: 
                reduce_loop = 0
                sumrescont_clus = np.zeros((len(uspis),realisations))
                Nummig_per_event = self.parameter_dictionary['Nm'][0]

            # Slow way to handle clusters - this should not be a problem if there are fewer than 100
            # In later updates this could potentially become another array index to sum over for greater efficiency: 
            # the problem would be unequal numbers of people between clusters...
            for i in range(0,len(uspis)):
        
                # If treatment has been specified, apply coverage fraction to individuals within a cluster (removing their worms completely)
                if self.treatment_times is not None:
                    if any(treat_ind) == True:

                        # Test if coverage pattern is specified
                        if self.treatment_coverages is not None:

                            # Generate random realisations of the treatment per individual
                            treatment_realisations = np.random.uniform(size=(np.sum(Nps[spis==uspis[i]]),realisations))

                            # Efficacy realisations of the treatment per individual
                            ws_after_treat = np.random.binomial(ws_ind_perclus[i].astype(int), \
                                             (1.0-self.drug_efficacy)*np.ones_like(ws_ind_perclus[i]),size=(np.sum(Nps[spis==uspis[i]]),realisations))

                            # Remove the worms of those treated
                            ws_ind_perclus[i] = ws_ind_perclus[i]*(treatment_realisations > cov_ind_perclus[i][np.arange(0,len(treat_ind),1)[treat_ind==True][0]]) + \
                                                ws_after_treat*(self.drug_efficacy < 1.0)*(treatment_realisations <= cov_ind_perclus[i][np.arange(0,len(treat_ind),1)[treat_ind==True][0]])

                        # Take into account the specified random compliance pattern if chosen
                        if self.compliance_params is not None:

                            # Obtain the alpha and beta values specified for the conditional probabilities of each individual (first round alpha entry is just the initial coverage probability)
                            alpha_val = comp_ind_perclus[i][np.arange(0,2*len(treat_ind),2)[treat_ind==True][0]]
                            beta_val = comp_ind_perclus[i][np.arange(0,2*len(treat_ind),2)[treat_ind==True][0]+1]
                            
                            # Generate random realisations of the treatment per individual
                            treatment_realisations = np.random.uniform(size=(np.sum(Nps[spis==uspis[i]]),realisations))

                            # Efficacy realisations of the treatment per individual
                            ws_after_treat = np.random.binomial(ws_ind_perclus[i].astype(int), \
                                             (1.0-self.drug_efficacy)*np.ones_like(ws_ind_perclus[i]),size=(np.sum(Nps[spis==uspis[i]]),realisations))

                            # If in first round, just remove the worms of those treated according to the coverage probability and store the past behaviour
                            if treat_ind[0] == True:
                                ws_ind_perclus[i] = ws_ind_perclus[i]*(treatment_realisations > alpha_val) + \
                                                    ws_after_treat*(self.drug_efficacy < 1.0)*(treatment_realisations <= alpha_val)
                                last_round_behaviour_ind_perclus[i] = (treatment_realisations <= alpha_val)

                            # If not in first round, compute probabilities for the individuals based on their last round behaviour and then apply treatment accordingly
                            if treat_ind[0] == False:

                                # Set the conditional probabilities
                                cond_probabilities = alpha_val*(last_round_behaviour_ind_perclus[i] == True) + beta_val*(last_round_behaviour_ind_perclus[i] == False)

                                # Remove the worms of those treated
                                ws_ind_perclus[i] = ws_ind_perclus[i]*(treatment_realisations > cond_probabilities) + \
                                                    ws_after_treat*(self.drug_efficacy < 1.0)*(treatment_realisations <= cond_probabilities)

                                # Store this round as the new 'last round behaviour'
                                last_round_behaviour_ind_perclus[i] = (treatment_realisations <= cond_probabilities)

                        # Output the exact time of treatment implementation unless otherwise suppressed
                        if self.suppress_terminal_output == False: print('Treatment ' + str(np.arange(1,len(treat_ind)+1,1)[treat_ind==True][0]) + \
                                                                         ' implemented: ' + str(np.round(time,2)) + ' years')

                        # Store the post-treatment prevalence in each cluster if the last treatment has just been performed
                        if treat_ind[len(treat_ind)-1] == True: treat_prevs_perclus.append(np.sum((ws_ind_perclus[i]>0),axis=0).astype(float)/ \
                                                                                           np.sum(Nps[spis==uspis[i]]).astype(float))

                # Worm uptake event rates
                urs = lam_ind_perclus[i]*FOIs_ind_perclus[i]*(FOIs_ind_perclus[i]>0.0)

                # Worm death event rates
                drs = (mus_ind_perclus[i]+mu1s_ind_perclus[i])*ws_ind_perclus[i].astype(float)

                # Total event rates
                trs = urs + drs + (np.ones((np.sum(Nps[spis==uspis[i]]),realisations))/do_nothing_timescale)
 
                # Call a unform-random number generator for the events available to the individual in the cluster
                randgen_ind_clus = np.random.uniform(size=(np.sum(Nps[spis==uspis[i]]),realisations))

                # Decide on worm uptake, death or nothing for the individual
                ws_ind_perclus[i] += (randgen_ind_clus < urs/trs) 
                ws_ind_perclus[i] -= (ws_ind_perclus[i]>0)*(randgen_ind_clus > urs/trs)*(randgen_ind_clus < (urs+drs)/trs)

                # Compute the total force of infection within the cluster and convert it into a matrix for calculation
                totFOI_clus = np.sum((1.0-(2.0**(1.0-ws_ind_perclus[i].astype(float))))*ws_ind_perclus[i].astype(float)* \
                                     np.exp(-gams_ind_perclus[i]*(ws_ind_perclus[i].astype(float)-1.0))/float(np.sum(Nps[spis==uspis[i]])),axis=0)
                totFOI_clus_mat = np.tensordot(np.ones(np.sum(Nps[spis==uspis[i]])),totFOI_clus,axes=0)

                # Update the forces of infection
                FOIs_ind_perclus[i] += ((mu2s_ind_perclus[i]*(mus_ind_perclus[i]+mu1s_ind_perclus[i])*R0s_ind_perclus[i]*totFOI_clus_mat) - \
                                        (mu2s_ind_perclus[i]*FOIs_ind_perclus[i]))*timestep

                # If migration has been specified, compute egg pulses into the reservoir
                if self.migration_mode == True:

                    # If specified, draw egg counts from distributions fixed to the initial ensemble mean worm burdens
                    if mf_migrations_fixed == True: last_ensM = Ms

                    # Migration event rate sum relative to each cluster - if the self-migration ([i,i] element) is chosen then half the sum of both migration rates
                    mig_rate_relclus = np.tensordot(((i==reduce_loop)*(rps[i][reduce_loop:] + rms[i][reduce_loop:])/2.0) + \
                                                    ((i!=reduce_loop)*(rps[i][reduce_loop:] + rms[i][reduce_loop:])) + \
                                                    (np.ones(len(uspis[reduce_loop:]))/do_nothing_timescale),np.ones(realisations),axes=0)
                    rps_mat = np.tensordot(rps[i][reduce_loop:],np.ones(realisations),axes=0)
                    rms_mat = np.tensordot(rms[i][reduce_loop:],np.ones(realisations),axes=0)

                    # Compute the egg pulse amplitudes from the mean field if specified
                    if mf_migrations == True:
                        # Egg pulse amplitude relative to each cluster computed from mean field
                        egg_pulse_relclus = np.asarray([self.egg_STH_pulse_sampler(last_ensM[spis==uspis[j]],j,realisations,Nummig_per_event) for j in range(reduce_loop,len(uspis))])

                    # Compute the egg pulse amplitudes from randomly-selected individuals in the respective reservoirs
                    if mf_migrations == False:
                        # Draw random egg counts from people for use in the standard reservoir pulses
                        egg_pulse_relclus = np.asarray([np.sum(worm_to_egg_func(ws_ind_perclus[j],gams_ind_perclus[j])[np.random.randint(0,np.sum(Nps[spis==uspis[j]]),\
                                                                                                  size=Nummig_per_event),:],axis=0) for j in range(reduce_loop,len(uspis))])

                    # Call a unform-random number generator for the migratory events available 
                    randgen_mig = np.random.uniform(size=(len(uspis[reduce_loop:]),realisations))
                    
                    # Sum over all possible reservoir contributions from each cluster and pulse eggs into reservoir
                    rescont_clus = ((randgen_mig < rps_mat/mig_rate_relclus)*np.ones((len(uspis[reduce_loop:]),realisations))*egg_pulse_relclus - \
                                   ((randgen_mig > rps_mat/mig_rate_relclus)*(randgen_mig < (rps_mat+rms_mat)/mig_rate_relclus)* \
                                     np.ones((len(uspis[reduce_loop:]),realisations))*np.tensordot(len(uspis[reduce_loop:]),egg_pulse_relclus[0],axes=0)))
                    sumrescont_clus[i] += np.sum(rescont_clus,axis=0)
                    rescont_clus_mat = ((mus_ind_perclus[i]+mu1s_ind_perclus[i])*R0s_ind_perclus[i]/float(np.sum(Nps[spis==uspis[i]])))* \
                                                                                 np.tensordot(np.ones(np.sum(Nps[spis==uspis[i]])),sumrescont_clus[i],axes=0)
                    eggpulse_ind_perclus[i] = rescont_clus_mat

                    # Determine migrations relative to the other clusters to be consistent with this one
                    if i < len(uspis)-1: sumrescont_clus[reduce_loop+1:len(uspis)] += -rescont_clus[1:]

                    # Reduce the loop size by one 
                    reduce_loop += 1

                # Compute the ensemble mean, ensemble variance as well as the upper and lower limits of the 68 confidence region in the mean worm burden per cluster 
                ensemble_of_m_perclus = np.sum(ws_ind_perclus[i].astype(float)/float(np.sum(Nps[spis==uspis[i]])),axis=0)
                ensM_perclus = np.sum(ensemble_of_m_perclus)/float(realisations)
                ensV_perclus = np.sum((ensemble_of_m_perclus-ensM_perclus)**2.0)/float(realisations)
                [ensup68CL_perclus,enslw68CL_perclus] = np.percentile(ensemble_of_m_perclus,[84,16])
                ensM_perclus_output += [ensM_perclus]
                ensV_perclus_output += [ensV_perclus]
                ensup68CL_perclus_output += [ensup68CL_perclus]
                enslw68CL_perclus_output += [enslw68CL_perclus]

                # If migration has been specified, update the age-binned ensemble mean worm burdens
                if self.migration_mode == True:
                    ws_age_binned = np.split(ws_ind_perclus[i].astype(float),Nps[spis==uspis[i]][:len(Nps[spis==uspis[i]])],axis=0)
                    last_ensM[uspis[i]==spis] = np.asarray([np.sum(ws_age_binned[j]/float(Nps[spis==uspis[i]][j]))/float(realisations) for j in range(0,len(Nps[spis==uspis[i]]))])

                # If treatment has been specified, compute the ensemble mean and ensemble variance in the mean worm burden per cluster
                # where the realisations which have been lost to the m(t) = 0 attractor post-treatment have been removed
                if self.treatment_times is not None:
                    ensM_zeros_removed_perclus = np.sum(ensemble_of_m_perclus*(ensemble_of_m_perclus>0.0))/float(np.sum((ensemble_of_m_perclus>0.0)))
                    ensV_zeros_removed_perclus = np.sum(((ensemble_of_m_perclus-ensM_perclus)**2.0)*(ensemble_of_m_perclus>0.0))/float(np.sum((ensemble_of_m_perclus>0.0)))
                    ensM_zeros_removed_perclus_output += [ensM_zeros_removed_perclus]
                    ensV_zeros_removed_perclus_output += [ensV_zeros_removed_perclus]

                # If migration has been specified, include egg pulses into the reservoir at the end of the integration step and reset the pulses
                if self.migration_mode == True: FOIs_ind_perclus[i] += eggpulse_ind_perclus[i]
        
            # Record the time, ensemble mean and ensemble variance as well as the upper and lower limits of the 68 confidence region in the mean worm burden per cluster in a list
            output_list = [time] + ensM_perclus_output + ensV_perclus_output + ensup68CL_perclus_output + enslw68CL_perclus_output
            
            # If treatment has been specified, add the ensemble mean and ensemble variance with the m(t) = 0 realisations removed per cluster to the output list
            if self.treatment_times is not None: output_list += ensM_zeros_removed_perclus_output + ensV_zeros_removed_perclus_output

            output_data.append(output_list)

            # Output a snapshot of the worm burdens in each cluster after each specified number of steps in time - filename contains time elapsed in years
            if len(timesteps_snapshot) != 0:
                if any(count_steps == tts for tts in timesteps_snapshot):

                    # Loop over each cluster
                    for i in range(0,len(uspis)):

                        # Output the data to a tab-delimited .txt file in the specified output directory
                        np.savetxt(self.path_to_helmpy_directory + '/' + self.output_directory + output_filename + '_snapshot_timestep_' + str(count_steps) + '_cluster_' + \
                                   str(uspis[i]) + '.txt', ws_ind_perclus[i].T, delimiter='\t')

                        # Due to Poissonian event draws, exact time of snapshot changes and is hence output for user records and comparison
                        if self.suppress_terminal_output == False: print('Output snapshot of worm burdens at time t = ' + str(np.round(time,2)) + ' years for cluster ' + str(uspis[i]))         

        if self.suppress_terminal_output == False: print('\n')

        # It treatment has been specified, output the post-last treatment realisations per cluster in specified file names
        if self.treatment_times is not None:

            # Output the post-last treatment realisations in each cluster
            for i in range(0,len(uspis)):

                # Output the data to tab-delimited .txt files in the specified output directory
                np.savetxt(self.path_to_helmpy_directory + '/' + self.output_directory + output_filename + '_lasttreat_prevalences_cluster_' + \
                           str(spis[i]) + '.txt',treat_prevs_perclus[i],delimiter='\t')

        # Output the final treatment realisations in each cluster
        for i in range(0,len(uspis)):   
            # Output the data to a tab-delimited .txt file in the specified output directory 
            np.savetxt(self.path_to_helmpy_directory + '/' + self.output_directory + output_filename + '_final_prevalences_cluster_' + \
                           str(spis[i]) + '.txt',np.sum((ws_ind_perclus[i]>0),axis=0).astype(float)/float(np.sum(Nps[spis==uspis[i]])),delimiter='\t')

        # Output the data to a tab-delimited .txt file in the specified output directory  
        np.savetxt(self.path_to_helmpy_directory + '/' + self.output_directory + output_filename + '.txt',output_data,delimiter='\t')


    # Define the mean-field worm sampler - draws stationary realisations of individual worm burdens from a cluster, stacked in age bins to match 
    # the full simulation - requires only the mean worm burdens in each age bin 'M', the spatial index number 'spi' and number of realiations 'size' 
    # where the remaining parameters are specified by the chosen ones in the cluster with the 'fix_groupings' tool
    def worm_STH_stationary_sampler(self,M,spi,size):

        # Fix the dimensions for all of the groupings if necessary
        self.fix_groupings()

        # Find spatial index number of grouping and the age-binned parameters
        spis = np.asarray(self.parameter_dictionary['spi'])

        # Find unique cluster references
        uspis = np.unique(spis)

        R0 = np.asarray(self.parameter_dictionary['R0'])[spis==uspis[spi]]
        k = np.asarray(self.parameter_dictionary['k'])[spis==uspis[spi]]
        gam = np.asarray(self.parameter_dictionary['gam'])[spis==uspis[spi]]
        N = np.asarray(self.parameter_dictionary['Np'])[spis==uspis[spi]] 

        # Define z
        z = np.exp(-gam)

        # Sum over egg counts distribution moments and variances 
        Eggfirstmom = np.sum(N.astype(float)*M*(((1.0+((1.0-z)*M/k))**(-k-1.0))-((1.0+((1.0-(z/2.0))*M/k))**(-k-1.0))))
        Eggsecondmom = np.sum(((N.astype(float))**2.0)*(((M+(((z**2.0)+(1.0/k))*(M**2.0)))/((1.0+((1.0-(z**2.0))*M/k))**(k+2.0))) + \
                              ((M+(((z**2.0/4.0)+(1.0/k))*(M**2.0)))/((1.0+((1.0-(z**2.0/4.0))*M/k))**(k+2.0))) - \
                              ((M+(((z**2.0)+(2.0/k))*(M**2.0)))/((1.0+((1.0-(z**2.0/2.0))*M/k))**(k+2.0)))))  
        Eggvariance = Eggsecondmom - (Eggfirstmom**2.0)

        # Draw realisations of an individual's uptake rate in each age bin
        k_inds = np.tensordot(np.asarray(np.concatenate([k[j]*np.ones(N[j]) for j in range(0,len(N))]).ravel()),np.ones(size),axes=0)      
        lam = np.random.gamma(k_inds,1.0/k_inds,size=(np.sum(N),size))

        # Draw stationary realisations of the force of infection sum from the reservoir 
        FOI = np.random.negative_binomial((Eggfirstmom**2.0)/np.abs(Eggvariance-Eggfirstmom),(Eggvariance>Eggfirstmom)*Eggfirstmom/Eggvariance,size=(np.sum(N),size))

        # Use approximate stationarity of reservoir to estimate Poisson walker intensity for each individual in each age bin
        R0_inds = np.tensordot(np.asarray(np.concatenate([R0[j]*np.ones(N[j]) for j in range(0,len(N))]).ravel()),np.ones(size),axes=0)
        Intensity = (lam*R0_inds*FOI/np.sum(N))

        # Draw N x realision Poissonian walkers to give the worm burdens
        samples = np.random.poisson(Intensity,size=(np.sum(N),size))
                
        # Output the worm burden samples 
        return samples


    # Define the egg pulse sampler, drawing the amplitude of "pulses" in egg count for the FOI going into or removed from a reservoir, from inter-cluster migration
    # and are assumed to require only the mean field parameters of the cluster to be determined - the function requires only the ensemble mean worm burden 'M', 
    # the spatial index number 'spi', number of realiations 'size' and the number of migrants per event as inputs where the remaining 
    # parameters are specified by the chosen ones in the cluster with 'fix_groupings'
    def egg_STH_pulse_sampler(self,M,spi,size,Nummig):

        # Fix the dimensions for all of the groupings if necessary
        self.fix_groupings()

        # Find spatial index number of grouping and the age-binned parameters
        spis = np.asarray(self.parameter_dictionary['spi'])

        # Find unique cluster references
        uspis = np.unique(spis)

        k = np.asarray(self.parameter_dictionary['k'])[spis==uspis[spi]]
        gam = np.asarray(self.parameter_dictionary['gam'])[spis==uspis[spi]]
        N = np.asarray(self.parameter_dictionary['Np'])[spis==uspis[spi]] 

        # Define z
        z = np.exp(-gam)

        # Obtain the number of age bins and draw random realisations of which age bin the egg pulse will be drawn from
        numagebins = len(N)
        reals = np.random.randint(0,numagebins,size=size) 

        # Sum over egg counts distribution moments and variances 
        Eggfirstmoms = (M*(((1.0+((1.0-z)*M/k))**(-k-1.0))-((1.0+((1.0-(z/2.0))*M/k))**(-k-1.0))))[reals]
        Eggsecondmoms = ((((M+(((z**2.0)+(1.0/k))*(M**2.0)))/((1.0+((1.0-(z**2.0))*M/k))**(k+2.0))) + \
                       ((M+(((z**2.0/4.0)+(1.0/k))*(M**2.0)))/((1.0+((1.0-(z**2.0/4.0))*M/k))**(k+2.0))) - \
                       ((M+(((z**2.0)+(2.0/k))*(M**2.0)))/((1.0+((1.0-(z**2.0/2.0))*M/k))**(k+2.0)))))[reals] 
        Eggvariances = Eggsecondmoms - (Eggfirstmoms**2.0)
        Eggfirstmom = np.tensordot(np.ones(Nummig),Eggfirstmoms,axes=0)
        Eggvariance = np.tensordot(np.ones(Nummig),Eggvariances,axes=0)

        # Draw stationary realisations of the force of infection sum from the reservoir 
        egg_pulse_samples = np.sum(np.random.negative_binomial((Eggfirstmom**2.0)/np.abs(Eggvariance-Eggfirstmom),(Eggvariance>Eggfirstmom)*Eggfirstmom/Eggvariance,size=(Nummig,size)),axis=0)

        # In case of invalid values, set samples to zero
        egg_pulse_samples[np.isnan(egg_pulse_samples)] = 0.0

        # Output an array of egg pulse sample realisations
        return egg_pulse_samples


    # Run the mean-field model and compute the ensemble mean and variance (does not implement treatment) and outputting to file
    def run_meanfield(self,
                      runtime,                  # Set the total time of the run in years
                      timestep,                 # Set a timestep to evolve the deterministic mean field
                      output_filename           # Set a filename for the data to be output in self.output_directory
                      ):
        
        # Terminal front page when code runs...
        if self.suppress_terminal_output == False: self.helmpy_frontpage()

        # Fix the dimensions for all of the groupings
        self.fix_groupings()

        if self.helm_type == 'STH':

            # Set parameter values, initial conditions and cluster references for each realisation
            mus = np.asarray(self.parameter_dictionary['mu'])
            mu1s = np.asarray(self.parameter_dictionary['mu1'])
            mu2s = np.asarray(self.parameter_dictionary['mu2'])
            R0s = np.asarray(self.parameter_dictionary['R0'])
            ks = np.asarray(self.parameter_dictionary['k'])
            gams = np.asarray(self.parameter_dictionary['gam'])
            Nps = np.asarray(self.parameter_dictionary['Np'])
            spis = np.asarray(self.parameter_dictionary['spi'])
            Ms = np.asarray(self.initial_conditions['M'])
            FOIs = np.asarray(self.initial_conditions['FOI'])
            zs = np.exp(-gams)

            # Find unique cluster references
            uspis = np.unique(spis)

            # Obtain the number of clusters
            numclus = len(uspis)

            if self.suppress_terminal_output == False: print('Setting initial conditions...')

            # Define the mean-field deterministic system of differential equations to govern the STH transmission dynamics
            def meanfield_STHsystem(time,MsFOIs):

                # Extract mean worm burdens and forces of infection and calculate first moment of egg count in the presence of sexual reproduction
                oldMs = MsFOIs[:int(len(MsFOIs)/2)]
                oldFOIs = MsFOIs[int(len(MsFOIs)/2):]
                oldfs = (1.0+((1.0-zs)*oldMs/ks))**(-ks-1.0)
                oldphis = 1.0 - (((1.0+((1.0-zs)*(oldMs)/ks))/(1.0+((2.0-zs)*(oldMs)/(2.0*ks))))**(ks+1.0))
                oldFOItots = np.asarray([np.sum((Nps.astype(float)*oldphis*oldfs*oldMs)[spi==spis])/np.sum(Nps[spi==spis].astype(float)) for spi in spis])

                # Use old values to compute new first derivatives in time for the mean field system in each cluster to evolve
                newMsderiv = oldFOIs - ((mus+mu1s)*oldMs)
                newFOIsderiv = mu2s*(((mus+mu1s)*R0s*oldFOItots)-oldFOIs)
                
                return np.append(newMsderiv,newFOIsderiv)   

            if self.suppress_terminal_output == False: 

                print('                       ')
                print('Total number of individuals: ' + str(np.sum(Nps)))
                print('Number of clusters: ' + str(numclus))
                print('                       ')
                print('Now running mean field model for ' + str(runtime) + ' years...')
                print('                                                                        ')

            count_steps = 0
            output_data = []
            time = 0.0 # Initialise time, mean worm burdens and forces of infection
            MsFOIs = np.append(Ms,FOIs)
            Ms0 = Ms
            FOIs0 = FOIs
            Integrandmean = [0.0 for spi in uspis]
            Integrandsecondmom = [0.0 for spi in uspis]
        
            # Loop over set timestep
            while time < runtime:

                # Update mean worm burdens and forces of infection with dynamics
                MsFOIs += meanfield_STHsystem(time,MsFOIs)*timestep
                Ms = MsFOIs[:int(len(MsFOIs)/2)]
                FOIs = MsFOIs[int(len(MsFOIs)/2):]  

                # Ensure no pathological values exist
                Ms = Ms*(Ms>0.0)
                FOIs = FOIs*(FOIs>0.0)

                # Sum over egg counts distribution moments and variances in each age bin
                Eggfirstmom = Nps.astype(float)*Ms*(((1.0+((1.0-zs)*Ms/ks))**(-ks-1.0))-((1.0+((1.0-(zs/2.0))*Ms/ks))**(-ks-1.0)))
                Eggsecondmom = ((Nps.astype(float))**2.0)*(((Ms+(((zs**2.0)+(1.0/ks))*(Ms**2.0)))/((1.0+((1.0-(zs**2.0))*Ms/ks))**(ks+2.0))) + \
                               ((Ms+(((zs**2.0/4.0)+(1.0/ks))*(Ms**2.0)))/((1.0+((1.0-(zs**2.0/4.0))*Ms/ks))**(ks+2.0))) - \
                               ((Ms+(((zs**2.0)+(2.0/ks))*(Ms**2.0)))/((1.0+((1.0-(zs**2.0/2.0))*Ms/ks))**(ks+2.0))))  
                Eggvariance = Eggsecondmom - (Eggfirstmom**2.0) 

                # Sum over egg counts distribution moments and variances per cluster
                SumEggfirstmom = [np.sum(Eggfirstmom[spi==spis]) for spi in uspis]
                SumEggsecondmom = [np.sum(Eggsecondmom[spi==spis]) for spi in uspis]
                SumEggvariance = [np.sum(Eggvariance[spi==spis]) for spi in uspis]

                # Compute the ensemble mean and variance of the sum of all individual mean worm burdens in each age bin
                Integrandmean = [(Integrandmean[spii]*np.exp(-(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])*timestep)) + \
                                 ((Nps[uspis[spii]==spis].astype(float)*(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])*SumEggfirstmom[spii]*R0s[uspis[spii]==spis]/ \
                                 np.sum(Nps[uspis[spii]==spis].astype(float)))*timestep) for spii in range(0,len(uspis))]

                Integrandsecondmom = [Nps[uspis[spii]==spis].astype(float)*(Ms[uspis[spii]==spis]**2.0)*np.exp(-2.0*(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])*time) - 
                                      2.0*Nps[uspis[spii]==spis].astype(float)*(Ms[uspis[spii]==spis]*(SumEggfirstmom[spii]* \
                                      R0s[uspis[spii]==spis]/np.sum(Nps[uspis[spii]==spis].astype(float))))* \
                                      (np.exp(-(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])*time)-np.exp(-2.0*(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])*time)) +
                                      ((Nps[uspis[spii]==spis].astype(float))*(1.0+(1.0/ks[uspis[spii]==spis]))* \
                                      (((1.0-np.exp(-(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])*time))/(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis]))**2.0)* \
                                      (((mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])**2.0)*(SumEggsecondmom[spii] + \
                                      SumEggvariance[spii])*(R0s[uspis[spii]==spis]**2.0)/(np.sum(Nps[uspis[spii]==spis].astype(float)))**2.0)) for spii in range(0,len(uspis))]

                ensmean = [(Nps[uspis[spii]==spis].astype(float)*Ms0[uspis[spii]==spis]* \
                           np.exp(-(mus[uspis[spii]==spis]+mu1s[uspis[spii]==spis])*time)) + Integrandmean[spii] for spii in range(0,len(uspis))]
                ensvariance = [ensmean[spii] + Integrandsecondmom[spii] - np.sum(Nps[uspis[spii]==spis].astype(float)*(Ms[uspis[spii]==spis]**2.0)) for spii in range(0,len(uspis))]

                # Update with specified timestep
                time += timestep

                # Count the number of steps performed in time
                count_steps += 1

                # Compute the normalised ensemble mean and ensemble variance in the mean worm burden per cluster using the inhomogenous Poisson solutions
                ensM_perclus_output = [np.sum(ensmean[spii])/np.sum(Nps[uspis[spii]==spis].astype(float)) for spii in range(0,len(uspis))]
                ensV_perclus_output = [np.sum(ensvariance[spii])/np.sum(Nps[uspis[spii]==spis].astype(float)**2.0) for spii in range(0,len(uspis))]

                # Record the time, ensemble mean and ensemble variance in the mean worm burden per cluster in a list
                output_list = [time]+ensM_perclus_output+ensV_perclus_output
                output_data.append(output_list)
                
            # Output the data to a tab-delimited .txt file in the specified output directory     
            np.savetxt(self.path_to_helmpy_directory + '/' + self.output_directory + output_filename + '.txt',output_data,delimiter='\t')


    # Run the mean-field stochastic model while computing the ensemble mean and ensemble variance as well as the upper and lower limits of the 68 confidence region and outputting to file
    def run_meanfield_stoch(self,
                            runtime,                    # Set the total time of the run in years
                            realisations,               # Set the number of stochastic realisations for the model
                            timestep,                   # Set a timestep to evolve the deterministic mean field
                            output_filename,            # Set a filename for the data to be output in self.output_directory
                            timesteps_snapshot=[],      # Optional - output a snapshot of the mean worm burdens in each cluster after a specified number of steps in time
                            ):
        
        # Terminal front page when code runs...
        if self.suppress_terminal_output == False: self.helmpy_frontpage()

        # Fix the dimensions for all of the groupings
        self.fix_groupings()

        if self.helm_type == 'STH':
            
            # Set parameter values, initial conditions and cluster references for each realisation
            mus = np.asarray(self.parameter_dictionary['mu'])
            mu1s = np.asarray(self.parameter_dictionary['mu1'])
            mu2s = np.asarray(self.parameter_dictionary['mu2'])
            R0s = np.asarray(self.parameter_dictionary['R0'])
            ks = np.asarray(self.parameter_dictionary['k'])
            gams = np.asarray(self.parameter_dictionary['gam'])
            Nps = np.asarray(self.parameter_dictionary['Np'])
            spis = np.asarray(self.parameter_dictionary['spi'])
            Ms = np.asarray(self.initial_conditions['M'])
            FOIs = np.asarray(self.initial_conditions['FOI'])
            zs = np.exp(-gams)

            # Find unique cluster references
            uspis = np.unique(spis)

            # Obtain the number of clusters
            numclus = len(uspis)

            # Obtain the number of groupings across all clusters - corresponding to the number of Langevin walkers
            numwalk = len(Nps)
            
            # Obtain an array of total number of people within the cluster of the indexed grouping (useful later)
            Nptots = np.asarray([np.sum(Nps[spis==spis[spii]]) for spii in range(0,len(spis))])

            if self.suppress_terminal_output == False: print('Setting initial conditions...')

            # Create realisations of the individual uptake rate sums 
            sumlams = np.asarray([np.sum(np.random.gamma(ks[i]*np.ones((realisations,Nps[i])), \
                                (1.0/ks[i])*np.ones((realisations,Nps[i])),size=(realisations,Nps[i])),axis=1) for i in range(0,numwalk)]).T

            # Initialise the force of infection and hence the summed Poisson intensities
            sumIntensityinit = np.tensordot(np.ones(realisations),(Nps*Ms).astype(float),axes=0)
            sumIntensityinteg = np.zeros((realisations,numwalk))
            sumIntensity = np.tensordot(np.ones(realisations),(Nps*Ms).astype(float),axes=0)
            FOI = np.tensordot(np.ones(realisations),FOIs,axes=0)

            # Define the transformation between fluctuations and mean worm burden locally in time
            def transform_to_mean_wb(xi_realisations,wb_realisations,inputs=[sumIntensityinteg,sumIntensity,FOI]):

                # Updates to the necessary values for iteration
                [sumIntensityinteg,sumIntensity,FOI] = inputs

                # Iterate the integral over time for the Poisson intensities
                sumIntensityinteg = (sumlams*FOI*timestep) + (sumIntensityinteg*np.exp(-np.tensordot(np.ones(realisations),(mus+mu1s),axes=0)*timestep))

                # Obtain the next intensity value
                sumIntensity = (sumIntensityinit*np.exp(-np.tensordot(np.ones(realisations),(mus+mu1s),axes=0)*time)) + sumIntensityinteg
                
                # Compute the transformation between variables - outputs [ new mean worm burden , summed intensity , the intensity integral]
                return [(sumIntensity + np.tensordot(np.ones(realisations), \
                         np.sqrt(Nps).astype(float),axes=0)*xi_realisations)/np.tensordot(np.ones(realisations),Nps.astype(float),axes=0), \
                         sumIntensity, sumIntensityinteg] 

            # Define the Langevin drift term for all walkers as a function
            def drift_function(xi,t):
                
                # Very important to initialise to zeros!!
                drifxit = np.zeros((realisations,numwalk)) 

                # Create the drift term contribution for the realisations of all walkers
                drifxit = np.tensordot(np.ones(realisations),-(mus+mu1s),axes=0)*xi

                # Output the drift term contribution
                return drifxit

            # Define the Langevin diffusion term for all walkers as a function
            def diffusion_function(xi,t,inputs=[sumIntensity,sumlams,FOI]):

                # Updates to the necessary values for iteration
                [sumIntensity,sumlams,FOI] = inputs

                # Very important to initialise to zeros!!
                diffxit = np.zeros((realisations,numwalk)) 

                # Create the diffusion term contribution for the realisations of all walkers
                diffxit = np.sqrt(((np.tensordot(np.ones(realisations),(mus+mu1s),axes=0)*sumIntensity)+(sumlams*FOI))/Nps)

                # Output the diffusion term contribution
                return diffxit 

            def Improved_Euler_Iterator(walker_nd,time,inputs=[sumIntensityinteg,sumIntensity,FOI]):
            # Iterate the solver with a strong order 1 Improved Euler Scheme from https://arxiv.org/abs/1210.0933

                # Updates to the necessary values for iteration
                [sumIntensityinteg,sumIntensity,FOI] = inputs

                random_number = np.random.normal(0.0,1.0,size=(len(walker_nd),len(walker_nd[0])))
                # Generate a random number for the Weiner process

                S_alternate = np.random.normal(0.0,1.0,size=(len(walker_nd),len(walker_nd[0])))
                # Generate a random number for alternator in Ito process

                K1 = (drift_function(walker_nd,time)*timestep) + \
                     (np.sqrt(timestep)*(random_number-(S_alternate/abs(S_alternate)))*diffusion_function(walker_nd,time,inputs=[sumIntensity,sumlams,FOI])) 
                K2 = (drift_function(walker_nd+K1,time+timestep)*timestep) + \
                     (np.sqrt(timestep)*(random_number+(S_alternate/abs(S_alternate)))*diffusion_function(walker_nd+K1,time+timestep,inputs=[sumIntensity,sumlams,FOI])) 

                return walker_nd + (0.5*(K1+K2)) 
                # Return next step from a group of realisations

            # Define the mean-field deterministic system of differential equations to govern the STH transmission dynamics
            def meanfield_STHsystem(time,MsFOIs):

                # Extract mean worm burdens and forces of infection and calculate first moment of egg count in the presence of sexual reproduction
                oldMs = MsFOIs[:int(len(MsFOIs)/2)]
                oldFOIs = MsFOIs[int(len(MsFOIs)/2):]
                oldfs = (1.0+((1.0-zs)*oldMs/ks))**(-ks-1.0)
                oldphis = 1.0 - (((1.0+((1.0-zs)*(oldMs)/ks))/(1.0+((2.0-zs)*(oldMs)/(2.0*ks))))**(ks+1.0))
                oldFOItots = np.asarray([np.sum((Nps.astype(float)*oldphis*oldfs*oldMs)[spi==spis])/np.sum(Nps[spi==spis].astype(float)) for spi in spis])

                # Use old values to compute new first derivatives in time for the mean field system in each cluster to evolve
                newMsderiv = oldFOIs - ((mus+mu1s)*oldMs)
                newFOIsderiv = mu2s*(((mus+mu1s)*R0s*oldFOItots)-oldFOIs)
                
                return np.append(newMsderiv,newFOIsderiv) 

            if self.suppress_terminal_output == False: 

                print('                       ')
                print('Total number of individuals: ' + str(np.sum(Nps)))
                print('Number of clusters: ' + str(numclus))
                print('                       ')
                print('Now running mean-field stochastic model for ' + str(runtime) + ' years...')
                print('                                                                        ')

            count_steps = 0
            output_data = []
            time = 0.0 # Initialise time, mean worm burden and fluctuation realisations
            walker_nd = (np.random.negative_binomial(np.tensordot(np.ones(realisations),Nps.astype(float)*ks,axes=0), \
                                                     np.tensordot(np.ones(realisations),((1.0+(Ms/ks))**(-1.0)),axes=0), \
                                                     size=(realisations,len(Nps))) - np.tensordot(np.ones(realisations),Nps.astype(float)*Ms,axes=0))/ \
                                                     np.tensordot(np.ones(realisations),np.sqrt(Nps.astype(float)),axes=0)
            mwb_walker_nd = np.tensordot(np.ones(realisations),Ms,axes=0)

            # Initialise quantities from the ensemble-averaged model in order to estimate the force of infection
            Ms0 = Ms
            FOIs0 = FOIs
            MsFOIs = np.append(Ms,FOIs)
            Integrandmean = [0.0 for spi in uspis]
            Integrandsecondmom = [0.0 for spi in uspis]  
 
            # Loop over set timestep
            while time < runtime:

                # Iterate mean worm burdens and forces of infection with dynamics 
                MsFOIs += meanfield_STHsystem(time,MsFOIs)*timestep

                # Ensure no pathological values exist
                MsFOIs = (MsFOIs>0.0)*MsFOIs

                # Update values
                Ms = MsFOIs[:int(len(MsFOIs)/2)]
                FOIs = MsFOIs[int(len(MsFOIs)/2):] 

                # Compute the force of infection estimator         
                FOIpois = sumlams*np.tensordot(np.ones(realisations),FOIs/Nps,axes=0)

                # Iterate the Langevin walkers using an Improved Euler method...
                walker_nd = Improved_Euler_Iterator(walker_nd,time,inputs=[sumIntensityinteg,sumIntensity,FOIpois])                 
                [mwb_walker_nd,sumIntensity,sumIntensityinteg] = \
                              transform_to_mean_wb(walker_nd,mwb_walker_nd,inputs=[sumIntensityinteg,sumIntensity,FOIpois])            

                # Make sure no negative solutions are obtained
                mwb_walker_nd = (mwb_walker_nd>0.0)*mwb_walker_nd

                # Update with specified timestep
                time += timestep

                # Count the number of steps performed in time
                count_steps += 1

                # Compute the normalised ensemble mean and ensemble variance as well as the upper and lower limits of the 68 confidence region
                # in the mean worm burden per cluster using the Langevin realisations
                meanwb_reals_perclus = [(np.sum(Nps[spis==uspis[spii]]*mwb_walker_nd[:,spis==uspis[spii]].astype(float),axis=1))/ \
                                         np.sum(Nps[spis==uspis[spii]].astype(float)) for spii in range(0,len(uspis))]
                ensM_perclus_output = [np.sum(meanwb_reals_perclus[spii])/float(realisations) for spii in range(0,len(uspis))]
                ensV_perclus_output = [(np.sum(meanwb_reals_perclus[spii]**2.0)/float(realisations)) - (ensM_perclus_output[spii]**2.0)  for spii in range(0,len(uspis))]    
                ensup68CL_perclus_output = [np.percentile(meanwb_reals_perclus[spii],84) for spii in range(0,len(uspis))]
                enslw68CL_perclus_output = [np.percentile(meanwb_reals_perclus[spii],16) for spii in range(0,len(uspis))]

                # Record the time, ensemble mean and ensemble variance as well as the upper and lower limits of the 68 confidence region in the mean worm burden per cluster in a list
                output_list = [time] + ensM_perclus_output + ensV_perclus_output + ensup68CL_perclus_output + enslw68CL_perclus_output
                output_data.append(output_list)

                # Output a snapshot of the mean worm burdens in each cluster after each specified number of steps in time - filename contains time elapsed in years
                if len(timesteps_snapshot) != 0:
                    if any(count_steps == tts for tts in timesteps_snapshot):

                        # Loop over each cluster
                        for i in range(0,len(uspis)):

                            # Output the data to a tab-delimited .txt file in the specified output directory
                            np.savetxt(self.path_to_helmpy_directory + '/' + self.output_directory + output_filename + '_snapshot_timestep_' + str(count_steps) + '_cluster_' + \
                                       str(uspis[i]) + '.txt', meanwb_reals_perclus[i], delimiter='\t')

                            # Due to Poissonian event draws, exact time of snapshot changes and is hence output for user records and comparison
                            if self.suppress_terminal_output == False: print('Output snapshot of worm burdens at time t = ' + str(np.round(time,2)) + ' years for cluster ' + str(uspis[i]))
                
            # Output the data to a tab-delimited .txt file in the specified output directory     
            np.savetxt(self.path_to_helmpy_directory + '/' + self.output_directory + output_filename + '.txt',output_data,delimiter='\t')


    # Just some front page propaganda...
    def helmpy_frontpage(self): 
        print('                                                         ')
        print('  >>>>      >>>>      >>>>      >>>>      >>>>      >>   ')
        print(' >>  >>    >>  >>    >>  >>    >>  >>    >>  >>    >> >> ')
        print('>>    >>>>>>    >>>>>>    >>>>>>    >>>>>>    >>>>>>   >>')
        print('>>                 >>                                    ')
        print('>>                 >>                                    ')
        print('>>         >>>>    >>      >>>    >>>    >>>>>>   >>   >>')
        print('>>>>>>>   >>  >>   >>     >> >>  >> >>  >>    >>  >>   >>') 
        print('>>    >>  >>  >>   >>     >>  >>>>  >>  >>    >>  >>   >>')
        print('>>    >>  >>>>>>   >>     >>   >>   >>  >>>>>>>   >>   >>')
        print('>>    >>  >>       >>     >>        >>  >>        >>   >>')
        print('>>    >>  >>   >>  >>     >>        >>  >>        >>   >>')
        print('>>    >>   >>>>>    >>>>> >>        >>  >>         >>>>>>')
        print('                                                      >> ')
        print('   >>>>      >>>>      >>>>      >>>>      >>>>      >>  ')
        print('  >>  >>    >>  >>    >>  >>    >>  >>    >>  >>    >>   ')
        print('>>>    >>>>>>    >>>>>>    >>>>>>    >>>>>>    >>>>>>    ')
        print('                                                         ')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('               Author: Robert J. Hardwick                ')
        print('              DISTRIBUTED UNDER MIT LICENSE              ')
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print('                                                         ')


