import sys
import signal
import traceback
import atexit
import os

import IPython
from IPython.core import ultratb, compilerop
from IPython.core.magic import Magics, magics_class, line_magic
from IPython.core.interactiveshell import DummyMod, InteractiveShell
from IPython.terminal.interactiveshell import TerminalInteractiveShell
from IPython.terminal.ipapp import load_default_config
from IPython.terminal.embed import InteractiveShellEmbed

def EXIT():
    os._exit(0)

def DBG(**kwargs):
    config = kwargs.get('config')
    header = kwargs.pop('header', u'')
    compile_flags = kwargs.pop('compile_flags', None)
    if config is None:
        config = load_default_config()
        config.InteractiveShellEmbed = config.TerminalInteractiveShell
        #HORIA
        config.InteractiveShell.confirm_exit = False
        config.TerminalIPythonApp.display_banner = False
        kwargs['banner1'] = '\nDBG>'        
        kwargs['config'] = config
        #HORIA        
    #save ps1/ps2 if defined
    ps1 = None
    ps2 = None
    try:
        ps1 = sys.ps1
        ps2 = sys.ps2
    except AttributeError:
        pass
    #save previous instance
    saved_shell_instance = InteractiveShell._instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
    frame = sys._getframe(1)
    shell = InteractiveShellEmbed.instance(_call_location_id='%s:%s' % (frame.f_code.co_filename, frame.f_lineno), **kwargs)
    shell(header=header, stack_depth=2, compile_flags=compile_flags)
    InteractiveShellEmbed.clear_instance()
    #restore previous instance
    if saved_shell_instance is not None:
        cls = type(saved_shell_instance)
        cls.clear_instance()
        for subclass in cls._walk_mro():
            subclass._instance = saved_shell_instance
    if ps1 is not None:
        sys.ps1 = ps1
        sys.ps2 = ps2
    EXIT()

def LOG(msg):
    print msg

def DBG_TOOLS_INIT():
    signal.signal(signal.SIGINT, signal_handler_sigint)
    atexit.register(signal_handler_exit)

def signal_handler_sigint(signal, frame):
    import signal
    print('')
    print('You pressed Ctrl+C!')
    print('-------------------------------------')
    traceback.print_stack()
    print('-------------------------------------')
    import IPython
    IPython.embed()

def signal_handler_exit():
    print('\nExecution finished.\n')
    print('-------------------------------------')
    DBG()
