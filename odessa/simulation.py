import numba as nb
import numpy as np
import json

from .phase import Phase
from .modules import index as modules_index

from .modules.Empty import Empty
from .helpers.meta import set_data, find_key
from .helpers.NumpyEncoder import NumpyEncoder

from scipy.integrate import solve_ivp


class Simulation:
    """The main class of ODESSA. This is where the simulations happen and results get returned.
    """
    def __init__(self):
        """List of the Simulation parameters

        Core -> The Core object that holds the variables needed for the simulation.
        phases -> A list that contains all the Phase objects that get run one after another.

        dt -> The MAXIMUM timestep allowed for a simulation
        tf -> The time after which the simulation stops regardless of terminator functions
        rtol -> The relative tolerance of the simulation
        atol -> The absolute tolerance of the simulation

        downsample -> The downsampling factor used when returning results
        method -> The method used by SciPy to solve the simulation. Full list here :
                  https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        
        config -> private attribute with the config dictionary, don't change this manually
                  use the fromJSON method instead. Should be made into a _private attribute

        sols -> List of n arrays containing the solutions. There is one array per phase.
        """
        self.Core = Empty()
        self.phases = []

        self.dt = np.inf
        self.tf = 1e3
        self.rtol = 1e-3
        self.atol = 1e-6
        self.downsample = 1
        
        self.method = None
        self.config = None
        
        self._history = {}
        self.sols = []

    @staticmethod
    def fromJSON(JSON):
        """Load a JSON and turns it into a Simulation object filled with Phase objects
           as well as metaparams such as dt, tf, etc.

        Args:
            JSON ([str]): A string containing a JSON file.

        Returns:
            [Simulation object]: The Simulation object ready to be run.
        """
        config = json.loads(JSON, strict=False)
        sim = Simulation()
        sim.readConfig(config)

        return sim

    def readConfig(self, config):
        """Takes a dictionary (generated from a JSON file) as an input and populates
           the Simulation object with all its parameters. This is called by fromJSON.

        Args:
            config ([dict]): The configuration dictionary
        """
        self.config = config

        # We start by setting the core from the config file
        self.Core = modules_index[config["Core"]["id"]]()
        set_data(self.Core, config["Core"])  # as well as initial conditions
        
        # set the meta paremters automatically
        set_data(self, config["Simulation"])

        # then we set the modules for all the phases
        for phase_config in config["phases"]:
            phase = Phase()
            phase.set_config(phase_config)
            self.phases.append(phase)

    @staticmethod
    @nb.njit
    def _rhs(t, y, Core, mod1, mod2, mod3, mod4, mod5, mod6,
             mod7, mod8, mod9, mod10):
        """ This function evaluates the right hand side composed of all the modules
            for a specific combination of time and a state variable.

            It resets the Core object, then fills it with the state variable passed as input,
            runs the modules serially and returns the dy attribute of the Core object.

        Args:
            t ([float]): The time
            y ([nparray of shape (m, )]): An array containg the m state variables 
            Core ([jitclass of Core type]): The Core module holding the state variables
            mod1 ([jitclass of Module type]): Module composed of jitclasses to be executed serially
            ...
            mod10 ([jitclass of Module type]): same as mod1, they're executed in order. 

        Returns:
            [nparray of shape (m, )]: returns an array containing 
                                      the derivative of the m state variables
        """
        Core.reset()
        Core.t = t
        Core.y = y

        # this is awkward but since all module objects are of different types
        # (they're not from the same class), numba refuses to put them in
        # the same list so they need to be passed one by one
        # this will need to be improved in the future

        mod1.rhs(Core)
        mod2.rhs(Core)
        mod3.rhs(Core)
        mod4.rhs(Core)
        mod5.rhs(Core)
        mod6.rhs(Core)
        mod7.rhs(Core)
        mod8.rhs(Core)
        mod9.rhs(Core)
        mod10.rhs(Core)

        return Core.dy

    def run(self):
        """Actually runs the simulation. Things happen in the following order :
        
        1. The Core is set with the initial conditions
        2. The modules in Phase 0 which have an on_start method run that method.
        3. Phase 0 is solved in solve_ivp. 
        4. The next phase is loaded where its initial conditions are the final conditions of the previous phase
        5. The next phase is solved.

        This process is repeated until the last phase is reached.
        After which the solution list is returned.
        
        Returns:
            [list]: A list of n arrays containing the solutions for the n previous phases. 
        """
        self.Core.y0 = self.Core.y
        print("ODESSA - On start methods...")
        for mod in self.phases[0].modlist:
            try:
                mod.on_start(self.Core) # check if the modules in the first phase have a on_start method
            except AttributeError:       # and run it if they do, otherwise skip to the next module
                continue

        y0 = self.Core.y

        for phase in self.phases:
            print("ODESSA - New phase")
            # input("new phase : "+str(np.degrees(self.Core.omega)))
            def rhs(t, y): return self._rhs(t, y, self.Core, *phase.modlist)
            sol = solve_ivp(rhs, [self.Core.t, self.tf], y0,
                            events=[event.event for event in phase.events],
                            max_step=self.dt, method=self.method, 
                            rtol=self.rtol, atol=self.atol)
            self.sols.append(sol)

            # print("Phase ended @ t={}s".format(sol.t[-1]))

            rhs(sol.t[-1], sol.y[:, -1])  # compute the last step again

            # extract the modified state from the last step to use
            # as initial step(bc for example tower updates quaternion)

            y0 = self.Core.y
            self.Core.t = sol.t[-1]
        return self.sols

    @property
    def JSON(self):
        """Returns a string containing a JSON object that can recreate the exact same simulation.
        Very useful for debugging. Get 'simulation.JSON' if your simulation crashes so you can recreate it.
        
        Returns:
            [string]: The string containing the full json. 
                      This can get pretty massive if you load big CSV tables in your JSON.
        """
        simDict = {}
        simDict["phases"] = []
        simDict["Core"] = {}
        simDict["Simulation"] = {"dt": self.dt,
                                 "tf": self.tf,
                                 "method": self.method,
                                 "atol" : self.atol,
                                 "rtol" : self.rtol}

        for phase in self.phases:
            simDict["phases"].append(phase.dump)

        index = {}
        for key in modules_index.keys():
            index[key] = modules_index[key]().__class__

        keyCore = find_key(index, self.Core.__class__)

        simDict["Core"]["id"] = keyCore
        simDict["Core"]["y"] = self.Core.y0

        return json.dumps(simDict, cls=NumpyEncoder, indent=2)

    def get_history(self):
        """Private method that gets the history including the state variable (pos, vel, etc.) and
           the auxiliary variables (speed of sound, thrust forces, etc.)

        Raises:
            ValueError: a ValueError gets raised if one tried to get the history before
                        actually running the simulation
        """ 
        self._history = {}

        if self.sols == []:
            raise ValueError("The list of solutions is empty - Run the simulation first.")

        # flushes all the dictionaries
        for phase in self.phases:
            for module in phase.modules.values():
                module.init_history()

        print("ODESSA - Getting logs...")
        self.Core.logging = True
        # run the rhs function over the solution with logging turned on
        for n, sol in enumerate(self.sols):
            modlist = self.phases[n].modlist
            for t, y in zip(sol.t[1::self.downsample], sol.y.T[1::self.downsample]): # skip the first value of the phase to avoid timestep duplication
                self._rhs(t, y, self.Core, *modlist) # XXX: Speed can probably be improved by wrapping
                                                                    # in a Numba call

        print("ODESSA - Processing logs...")
        # create a temporary history for each phase, that then fills in the full history
        for phase in self.phases:
            temp_history = {}
            for module in phase.modules.values():
                temp_history.update(module.history)

            if len(temp_history['t']) == 2:
                continue

            for key in temp_history.keys():
                try:
                    self._history[key] = np.append(self._history[key], temp_history[key][1:])  # first value is a zero so we pop it out
                except KeyError:
                    self._history[key] = temp_history[key][1:]

            full_keys = list(self._history.keys())
            temp_keys = list(temp_history.keys())

            keys = [key for key in full_keys if key not in temp_keys]

            for key in keys:
                self._history[key] = np.append(self._history[key], np.zeros(temp_history["t"].shape)[1:])
        print("ODESSA - Done!")
        self.Core.logging = False

    @property
    def history(self):
        """Property actually used by the user to reach the history.

        Returns:
            [dictionary]: All the variables are stored as 1d Numpy arrays. 
                          The key in the dictionary are the variable names, and the
                          values the actual history.
        """
        self.get_history()        
        return self._history

    def reset(self):
        """Resets the simulation including the solutions, the core and the history.
        """
        self.sols = []
        self.Core.reset()
        self._history = {}