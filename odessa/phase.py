from .modules.Empty import Empty as EmptyModule
from .events.Empty import Empty as EmptyEvent

from .modules import index as indexModules
from .events import index as indexEvents
from .helpers.meta import set_data, find_key

from collections import OrderedDict
import inspect


class Phase:
    """The Phase object is a datastructure that holds a list of 
    modules as well as terminator events for that Phase.

    It also contains the necessary logic for loading parameters in the modules.
    """
    def __init__(self):
        self.modules = OrderedDict()

        self.events = [EmptyEvent()]  # contains an event that never triggers by default

    def set_config(self, phase_config):
        """ Uses the phase configuration dictionary to load modules 
            and set their parameters

        Args:
            phase_config ([dict]): The phase dictionary loaded from the JSON
        """
        # iterates through all the modules
        for module_key in phase_config["modules"].keys():
            try:
                # create an object with the key
                module = indexModules[module_key]()
            except KeyError as err:
                print("WARNING - Failed setting module with id {}\
                    \n {}".format(module_key, err))
                continue
            # gets the dict defining the module in the config file
            module_definition = phase_config["modules"][module_key]

            # because they all have id we can lookup which module corresponds
            # in the module index to the one in the config file
            # we set the extra parameters it might need
            try:
                set_data(module, module_definition)
            except AttributeError as err:
                print("WARNING in {} : {}".format(module.id, err))

            # in the phase object we set the module to
            # the one wanted by the config file
            self.modules[module.id] = module

        # now that we're done with module we need the
        # phase switching event
        events_config = phase_config["events"]
        for event_id in events_config.keys():
            # we get the generator object for our event
            generator = indexEvents[event_id]()
            config = events_config[event_id]

            try:
                set_data(generator, config)
            except AttributeError as err:
                print("WARNING in {} : {}".format(generator.id, err))
            # we set event terminality and direction

            # then we add it to the events list
            self.events.append(generator)

    @property
    def modlist(self):
        """Returns a list of 10 modules.
            If the phase contains less than 10 modules, it fills the rest with EmptyModule

            This is needed bc the RHS uses a fixed amount of module regardless of the phase.

        Returns:
            [type]: [description]
        """
        return list(self.modules.values()) + [EmptyModule() for i in range(10-len(self.modules.keys()))]

    def addModule(self, object):
        """Function that adds a module to the phase

        Args:
            object ([jitclass]): The jitclass module to be added.
        """
        self.modules[object.id] = object

    def removeModule(self, id):
        """Removes a module from the phase

        Args:
            id ([str]): The id property of the module you want to remove.
        """
        del self.modules[id]

    @property
    def dump(self):
        """ Returns a dictionary config file from the Phase.

        This is VERY useful in order to generate a JSON when the simulation crashe
        that makes it easily replicable.

        Returns:
            [dict]: A dictionary that defines the Phase modules, and their parameters.
        """
        phasedict = {}
        phasedict["modules"] = {}
        phasedict["events"] = {}

        # get an index we can work with to find modules
        index = {}
        for key in indexModules.keys():
            index[key] = indexModules[key]().__class__

        # go through all the object modules we have
        for module in self.modules.values():
            # find their key in the general module index
            key_module = find_key(index, module.__class__)

            # create a dict with that key
            phasedict["modules"][key_module] = {}

            # go through every attributes of the module, removing generic shit and the rhs method
            varlist = [a for a in dir(module) if not a.startswith(
                '_') and a not in ['history', 'id', 'type', 'rhs', 'init_history', 'on_start']]

            # adds a key for each attribute with their value in the object as a value
            for var in varlist:
                phasedict["modules"][key_module][var] = getattr(module, var)

        # repeat the process for events
        for event in self.events:
            key_event = find_key(indexEvents, event.__class__)

            if key_event == 'Empty':
                continue

            phasedict["events"][key_event] = {}

            varlist = [a for a in dir(event) if not a.startswith(
                '_') and a not in ['event']]

            for var in varlist:
                phasedict["events"][key_event][var] = getattr(event, var)

        return phasedict
