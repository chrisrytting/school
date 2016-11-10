def unimodal(tosort):
    """
    tosort: list of unimodal sequence to find maximum of.
    """
    half1 = tosort[:(len(tosort)/2)]
    half2 = tosort[(len(tosort)/2):]
    if len(tosort) == 1:
        return tosort
    else:
        if half1[-1] > half2[0]:
            return unimodal(half1) 
        elif half1[-1] < half2[0]:
            return unimodal(half2)

unimodalsequence = [1,2,3,4,17,8,2,1]
print unimodal(unimodalsequence)

