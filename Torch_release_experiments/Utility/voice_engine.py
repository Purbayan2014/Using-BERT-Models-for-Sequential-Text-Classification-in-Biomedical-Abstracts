import pyttsx3

def _start_operation():
    print('Initiating')

def _exec_operation(operation, location, length_task):
    print('operation', operation, location, length_task)

def _end_operation(operation, ended):
    print('ending', operation, ended)

def vc_arch(task):
    vc_architecture = pyttsx3.init()
    vc_architecture.setProperty('rate', 160)
    vc_architecture.connect('Started-operation', _start_operation)
    vc_architecture.connect('executing-operation', _exec_operation)
    vc_architecture.connect('ending-operation', _end_operation)


    vc_architecture.say(task)
    vc_architecture.runAndWait()