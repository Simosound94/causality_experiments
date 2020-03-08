# -*- coding: utf-8 -*-
import nauka
import io, logging, os, sys, time, pdb
import torch
import uuid

from   PIL                              import Image
from   zipfile                          import ZipFile

from   .models                          import *


class ExperimentBase(nauka.exp.Experiment):
    """
    The base class for all experiments.

    NOTE: This class inherits from nauka.exp.Experiment. 
    An experiment comprises both an in-memory state and an on-disk state. At
	regular intervals, the in-memory state is synchronized with the on-disk
	state, thus permitting a resume should the experiment be killed. These
	on-disk serializations are called "snapshots".

    some of the methods are:
     load(self, path): Load state of the experiment from given path.
     dump(self, path): Dump state to the directory path
     fromScratch(self): Start an experiment from a snapshot
     snapshot(self) : Take a snapshot of the experiment (uses dump(self, path) )
     purge(self, ...): Purge snapshot directory of all the snapshots preserving
                       only some of those (e.g. the last one)


    """
    def __init__(self, a):
        self.a = type(a)(**a.__dict__)
        self.a.__dict__.pop("__argp__", None)
        self.a.__dict__.pop("__argv__", None)
        self.a.__dict__.pop("__cls__",  None)
        if self.a.workDir:
            super().__init__(self.a.workDir)
        else:
            projName = "CausalOptimization-40037046-a359-470b-b327-af9bbef3e532"
            expNames = [] if self.a.name is None else self.a.name
            workDir  = nauka.fhs.createWorkDir(self.a.baseDir, projName, self.uuid, expNames)
            super().__init__(workDir)
        self.mkdirp(self.logDir)
    
    def reseed(self, password=None):
        """
        Reseed PRNGs for reproducibility at beginning of interval.
        """
        password = password or "Seed: {} Interval: {:d}".format(self.a.seed,
                                                                self.S.intervalNum,)
        nauka.utils.random.setstate           (password)
        nauka.utils.numpy.random.set_state    (password)
        nauka.utils.torch.random.manual_seed  (password)
        nauka.utils.torch.cuda.manual_seed_all(password)
        return self
    
    def brk(self, it, max=None):
        """ Iterate through an iterator. Ends in case of debug or max reached
        """
        for i, x in enumerate(it):
            if self.a.fastdebug and i>=self.a.fastdebug: break
            if max is not None  and i>=max:              break
            yield x
    
    @property
    def uuid(self):
        u = nauka.utils.pbkdf2int(128, self.name)
        u = uuid.UUID(int=u)
        return str(u)
    @property
    def dataDir(self):
        return self.a.dataDir
    @property
    def logDir(self):
        return os.path.join(self.workDir, "logs")
    @property
    def isDone(self):
        return (self.S.epochNum >= self.a.num_epochs or
               (self.a.fastdebug and self.S.epochNum >= self.a.fastdebug))
    @property
    def exitcode(self):
        return 0 if self.isDone else 1



class Experiment(ExperimentBase):
    """
    Causal experiment.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        class MsgFormatter(logging.Formatter):
            def formatTime(self, record, datefmt):
                t           = record.created
                timeFrac    = abs(t-int(t))
                timeStruct  = time.localtime(record.created)
                timeString  = ""
                timeString += time.strftime("%F %T", timeStruct)
                timeString += "{:.3f} ".format(timeFrac)[1:]
                timeString += time.strftime("%Z",    timeStruct)
                return timeString
        formatter = MsgFormatter("[%(asctime)s ~~ %(levelname)-8s] %(message)s")
        handlers = [
            logging.FileHandler  (os.path.join(self.logDir, "log.txt")),
            logging.StreamHandler(sys.stdout),
        ]
        for h in handlers:
            h.setFormatter(formatter)
        logging.basicConfig(
            level    = logging.INFO,
            handlers = handlers,
        )
        logging.info("*****************************************")
        logging.info("Command:             "+" ".join(sys.argv))
        logging.info("CWD:                 "+os.getcwd())
        logging.info("Experiment Name:     "+self.name)
        logging.info("Experiment UUID:     "+self.uuid)
        logging.info("Experiment Work Dir: "+self.workDir)
        logging.info("")
        logging.info("")
    
    @property
    def name(self):
        """A unique name containing every attribute that distinguishes this
        experiment from another and no attribute that does not."""
        attrs = [
            self.a.seed,
            self.a.model,
            self.a.num_epochs,
            self.a.batch_size,          #                                                                 |  DEFAULT
            self.a.dpe,                 # (int) Nr of training distribution per epoch                     |  1000
            self.a.train_functional,    # (int) Nr of batches for functional parameters per distribution  |  0
            self.a.ipd,                 # (int) Nr of interventions per distribution                      |  100
            self.a.hidden_truth,        # (int) Nr of hidden neurons in ground-truth network.             |  None
            self.a.hidden_learn,        # (int) Nr of hidden neurons in learner network.                  |  None
            self.a.num_vars,            # (int) Nr of variables in the system (M in paper)                |  5
            self.a.num_cats,            # (int) Nr of categories per variable (N in paper)                |  3
            self.a.num_parents,         # (int) Nr of expected parents. Default is 5.                     |  5
            self.a.cpi,                 # (int) Nr Configurations per intervention                        |  20
            self.a.xfer_epi_size,       # (int) Nr Transfer episode size                                  |  10
            self.a.predict,             # (int) Nr of iterations to predict which node was intervened upon|  0
            self.a.predict_cpb,         # (int) Configurations per batch during intervention prediction   |  10
            self.a.temperature,         # (float) Temperature of the MLP.                                 |  1.0
            self.a.structural_only,     # (bool) whether to learn structural parameters only              |  False
            self.a.structural_init,     # (bool) Initialize structural parameters to ground truth         |  False
            self.a.graph,               # (str) structure of the causal graph (ground thruth)             |  None
            self.a.cuda,                #                                                                 |  
            self.a.model_optimizer,     # (str) Model Optimizer                                           |  nag:0.001,0.9 
            self.a.gamma_optimizer,     # (str) Gamma optimizer                                           |  nag:0.0001,0.9
            self.a.lmaxent,             # (float) Regularizer for maximum entropy                         |  0.00 
            self.a.lsparse,             # (float) Regularizer for sparsity                                |  0.00
            self.a.ldag,                # (float) Regularizer for DAGness                                 |  0.1
            self.a.fastdebug,           # (bool) Debug                                                    |  False
        ]
        return "-".join([str(s) for s in attrs]).replace("/", "_")
    
    def load(self, path):
        self.S = torch.load(os.path.join(path, "snapshot.pkl"))
        return self
    
    def dump(self, path):
        torch.save(self.S,  os.path.join(path, "snapshot.pkl"))
        return self
    
    def fromScratch(self):
        pass
        """Reseed PRNGs for initialization step"""
        self.reseed(password="Seed: {} Init".format(self.a.seed))
        
        """Create snapshottable-state object"""
        self.S = nauka.utils.PlainObject()
        
        """Model Instantiation"""
        self.S.model = None
        if   self.a.model == "cat":
            self.S.model  = CategoricalWorld(self.a)
        elif self.a.model == "asia":
            self.S.model  = AsiaWorld(self.a)
        if   self.S.model is None:
            raise ValueError("Unsupported model \""+self.a.model+"\"!")
        
        if self.a.cuda:
            self.S.model  = self.S.model.cuda(self.a.cuda[0])
        else:
            self.S.model  = self.S.model.cpu()
        
        """Optimizer Selection"""
        self.S.msoptimizer = nauka.utils.torch.optim.fromSpec(self.S.model.parameters_slow(),       self.a.model_optimizer)
        self.S.goptimizer  = nauka.utils.torch.optim.fromSpec(self.S.model.structural_parameters(), self.a.gamma_optimizer)
        
        """Counters"""
        self.S.epochNum    = 0
        self.S.intervalNum = 0
        self.S.stepNum     = 0
        
        return self
    
    def run(self):
        """Run by intervals until experiment completion."""
        while not self.isDone:
            self.interval().snapshot().purge()
        return self
    
    def interval(self):
        """
        An interval is defined as the computation- and time-span between two
        snapshots.
        
        Hard requirements:
        - By definition, one may not invoke snapshot() within an interval.
        - Corollary: The work done by an interval is either fully recorded or
          not recorded at all.
        - There must be a step of the event logger between any TensorBoard
          summary log and the end of the interval.
        
        For reproducibility purposes, all PRNGs are reseeded at the beginning
        of every interval.
        """
        
        self.reseed()
        
        
        """Training Loop"""
        self.S.model.train()
        for q in self.brk(range(self.a.dpe)): # For each training distribution in epoch
            if q>0: self.S.stepNum += 1
            
            # ==================================================================
            # 0) Initialize a new ground truth model with the same causal graph
            # ==================================================================
            self.S.model.alterdists()           # reinitialize randomly the weights of the ground truth model
                                                # (but keep the struct of the causal graph ( gammagt ) unchanged)
            self.S.model.zero_fastparams()      # Set to zero the fast parameters of the learner
            
            
            # ==================================================================
            # 1) Train slow parameters only Loop (to adapt to new ground truth model)
            # ==================================================================
            if self.a.train_functional:
                smpiter = self.S.model.sampleiter(self.a.batch_size) # An iterator of batch_size samples from the ground truth model
                cfgiter = self.S.model.configpretrainiter()          # An iterator of causal structures drawn from the learner parameters gamma

                # Train the functional parameters
                for b, (batch, config) in self.brk(enumerate(zip(smpiter, cfgiter)), max=self.a.train_functional):
                    self.S.msoptimizer.zero_grad()                          # self.S.msoptimizer optimizes only the slow parameters of the learner net
                    # NLL: negative log likelihood
                    nll = -self.S.model.logprob(batch, config)[0].mean()    # Compute the loss from the learner on the batch with that config (using fast + slow params)
                    nll.backward()                                          # Compute the gradients
                    self.S.msoptimizer.step()                               # Train the slow parameters of the learner network
                    if self.a.verbose and b % self.a.verbose == 0:
                        logging.info("Train functional param only NLL: "+str(nll.item()))
            
            # ==================================================================
            # 2) Interventions Loop
            #    2.1) An intervention is done
            #    2.2) Estimate the node upon which the intervention was made
            #    2.3) Compute the loss / gradients w.r.t the intervention node [TODO NON L'HO CAPITA DEL TUTTO]
            #    2.4) Adapt the fast parameters of the learner to the intervention
            #    2.5) Optimize gamma
            # ==================================================================
            for j in self.brk(range(self.a.ipd)):
                if j>0: self.S.stepNum += 1
                intervention_tstart = time.time()
                
                """Perform intervention under guard."""
                # Perform an intervention which modifies the ground truth model at the beginning fo the loop
                # and undo it at the end of the loop

                # ==============================================================
                # 2.1) An intervention is done
                # ==============================================================
                with self.S.model.intervene() as intervention:  

                    # ==========================================================
                    # 2.2) Estimate the node upon which the intervention was made
                    # ==========================================================
                    """Possibly attempt to predict the intervention node,
                       instead of relying on knowledge of it."""
                    if self.a.predict:
                        with torch.no_grad():
                            accnll  = 0
                            smpiter = self.S.model.sampleiter(self.a.batch_size) # An iterator of batch_size samples from the ground truth model
                            cfgiter = self.S.model.configpretrainiter()          # An iterator of causal structures drawn from the learner parameters gamma

                            # Use self.a.predict batches and select the node with the smallest NLL
                            # as the one for which the intervention was made. 

                            # TODO [Simone M.]: It might be better to select the node where the difference
                            # NLL_after_intervention - NLL_before_intervention is the biggest ?

                            for batch in self.brk(smpiter, max=self.a.predict):             # Average result on self.a.predict batches
                                for config in self.brk(cfgiter, max=self.a.predict_cpb):    # Average the result on self.a.predict_cpb causal structures drawn
                                    accnll += -self.S.model.logprob(batch, config)[0].mean(0)
                            selnode = torch.argmax(accnll).item()
                            logging.info("Predicted Intervention Node: {}  Actual Intervention Node: {}".format([selnode], list(iter(intervention))))
                            intervention = selnode
                    
                    self.S.goptimizer.zero_grad()
                    self.S.model.gamma.grad = torch.zeros_like(self.S.model.gamma)
                    
                    # ==========================================================
                    # 2.3) Compute the loss / gradients [TODO NON L'HO CAPITA DEL TUTTO]
                    # ==========================================================                    

                    gammagrads = [] # List of T tensors of shape (M,M,) indexed by (i,j)
                    logregrets = [] # List of T tensors of shape (M,)   indexed by (i,)
                    
                    """Transfer Episode Adaptation Loop"""
                    smpiter = self.S.model.sampleiter(self.a.batch_size) # An iterator of batch_size samples from the ground truth model
                    for batch in self.brk(smpiter, max=self.a.xfer_epi_size):
                        gammagrad = 0
                        logregret = 0
                        
                        """Configurations Loop"""
                        cfgiter = self.S.model.configiter()             # An iterator of causal structures drawn from the learner parameters gamma
                        for config in self.brk(cfgiter, max=self.a.cpi):# For each intervention, consider self.a.cpi configurations
                            """Accumulate Gamma Gradient"""
                            if self.a.predict:                          # If the intervention node has been estimated
                                # Compute the loss only for the intervention node [TODO NON NE SONO SICURO!]
                                logpn, logpi = self.S.model.logprob(batch, config, block=intervention)
                            else:
                                # Compute the loss for all nodes
                                logpn, logpi = self.S.model.logprob(batch, config)
                            with torch.no_grad():

                                # TODO: Why are gradients computed in this way?
                                gammagrad += self.S.model.gamma.sigmoid() - config
                                logregret += logpn.mean(0)
                            logpi.sum(1).mean(0).backward()             # Compute the gradients
                        
                        gammagrads.append(gammagrad)
                        logregrets.append(logregret)
                    

                    # ==========================================================
                    # 2.4) Adapt the fast parameters of the learner to the intervention
                    # ==========================================================
                    """Update Fast Optimizer"""
                    # Train the fast parameters of the learner with the model_optimizer
                    # to adapt to the interventon
                    for batch in self.brk(smpiter, max=self.a.xfer_epi_size):
                        self.S.model.zero_fastparams()
                        self.S.mfoptimizer = nauka.utils.torch.optim.fromSpec(
                            self.S.model.parameters_fast(), self.a.model_optimizer)
                        self.S.mfoptimizer.zero_grad()
                        cfgiter = self.S.model.configiter()
                        for config in self.brk(cfgiter, max=self.a.cpi):
                            logprob = self.S.model.logprob(batch, config)[0].sum(1).mean()
                            logprob.backward()
                        self.S.mfoptimizer.step()
                    all_logprobs = []
                    for batch in self.brk(smpiter, max=self.a.xfer_epi_size):
                        cfgiter = self.S.model.configiter()
                        for config in self.brk(cfgiter, max=self.a.cpi):
                            all_logprobs.append(self.S.model.logprob(batch, config)[0].mean())
                    
                    # ==========================================================
                    # 2.5) Optimize gamma
                    # ==========================================================
                    """Gamma Gradient Estimator"""
                    with torch.no_grad():
                        gammagrads = torch.stack(gammagrads)
                        logregrets = torch.stack(logregrets)
                        normregret = logregrets.softmax(0)

                        # R is the meta-objective (loss) for the slow params [see paper]
                        # dRdgamma are its gradients w.r.t. gamma
                        dRdgamma   = torch.einsum("kij,ki->ij", gammagrads, normregret)
                        self.S.model.gamma.grad.copy_(dRdgamma)
                        all_logprobs = torch.stack(all_logprobs).mean()
                    
                    """Gamma Regularizers"""
                    siggamma = self.S.model.gamma.sigmoid()
                    Lmaxent  = ((siggamma)*(1-siggamma)).sum().mul(-self.a.lmaxent)
                    Lsparse  = siggamma.sum().mul(self.a.lsparse)
                    Ldag     = siggamma.mul(siggamma.t()).cosh().tril(-1).sum() \
                                       .sub(self.S.model.M**2 - self.S.model.M) \
                                       .mul(self.a.ldag)
                    (Lmaxent + Lsparse + Ldag).backward()
                    
                    """Perform Gamma Update with constraints"""
                    self.S.goptimizer.step()
                    self.S.model.reconstrain_gamma()
                    
                    """Stop timer"""
                    intervention_tend = time.time()
                    
                    """Print the state of training occasionally"""
                    if self.a.verbose:
                        with torch.no_grad():
                            # Compute Binary Cross-Entropy over gammas, ignoring diagonal
                            siggamma  = self.S.model.gamma.sigmoid()
                            pospred   = siggamma.clone()
                            negpred   = 1-siggamma.clone()
                            posgt     = self.S.model.gammagt
                            neggt     = 1-self.S.model.gammagt
                            pospred.diagonal().fill_(1)
                            negpred.diagonal().fill_(1)
                            bce       = -pospred.log2_().mul_(posgt) -negpred.log2_().mul_(neggt)
                            bce       = bce.sum()
                            bce.div_(siggamma.numel() - siggamma.diagonal().numel())
                            
                            logging.info("")
                            logging.info("**************************")
                            logging.info("Gamma GT:   "+os.linesep+str(self.S.model.gammagt.detach()))
                            logging.info("Gamma:      "+os.linesep+str(siggamma))
                            logging.info("dRdGamma:   "+os.linesep+str(dRdgamma))
                            logging.info("Gamma Grad: "+os.linesep+str(self.S.model.gamma.grad.detach()))
                            logging.info("Gamma CE:   "+str(bce.item()))
                            logging.info("Intervention Time (s):       "+str(intervention_tend-intervention_tstart))
                            logging.info("Exp. temp. Transfer logprob: "+str(all_logprobs.item()))
                            logging.info("")
                            
                            if self.S.stepNum % self.a.verbose == 0:
                                # Append a PNG to a Zip file to avoid too many files
                                # on the filesystem
                                GAMMABIO = io.BytesIO()
                                GAMMAVIZ = self.S.model.vizualize_gamma().numpy()
                                GAMMAIMG = Image.fromarray(GAMMAVIZ, "RGB")
                                GAMMAIMG.save(GAMMABIO, "png")
                                GAMMAPNG = "gamma-{:07d}.png".format(self.S.stepNum)
                                GAMMAZIP = os.path.join(self.logDir, "gamma.zip")
                                with ZipFile(GAMMAZIP, 'a') as GAMMAZIP:
                                    GAMMAZIP.writestr(GAMMAPNG, GAMMABIO.getvalue())
        
        
        """Exit"""
        logging.info("Epoch {:d} done.\n".format(self.S.epochNum))
        self.S.epochNum    += 1
        self.S.intervalNum += 1
        self.S.stepNum     += 1
        return self
