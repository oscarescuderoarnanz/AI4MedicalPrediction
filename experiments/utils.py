# Libraries


# -----------------------------------------------------------
#                        Shortcuts
# -----------------------------------------------------------
# .. note: There is an option. of passing the variables using
#          **locals(). However, this will also pass variables
#          which have been created within the function.

def sep(*args, **kwargs):
    """"""
    return sepsis(*args, **kwargs)

def bac(args, **kwargs):
    """"""
    return bacteremia(*args, **kwargs)

def bsi(args, **kwargs):
    """"""
    return bloodstream_infection(*args, **kwargs)



def sepsis(a=1, b=2, c=3):
    """"""
    print(a, b, c)
    pass

def bacteremia():
    """"""
    pass

def bloodstream_infection():
    """"""
    pass