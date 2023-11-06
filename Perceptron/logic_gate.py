def AND(x1, x2):
    if x2 > -x1 + 1.5: y = 1
    else: y = 0
    return y

def OR(x1, x2):
    if x2 > -x1 + 0.5: y = 1
    else: y = 0
    return y

def NAND(x1, x2):
    if x2 < -x1 + 1.5: y = 1
    else: y = 0
    return y

def XOR(x1, x2):
    p = NAND(x1, x2)
    q = OR(x1, x2)
    y = AND(p, q)
    return y

def half_adder(A, B):
    S = XOR(A, B)
    C = AND(A, B)
    return S, C

def full_adder(A, B, Cin):
    P = XOR(A, B)
    S = XOR(P, Cin)
    Q = AND(P, Cin)
    R = AND(A, B)
    Cout = OR(Q, R)
    return S, Cout

def adder4(A, B):
    S0, C0 = half_adder(A[-1], B[-1])
    S1, C1 = full_adder(A[-2], B[-2], C0)
    S2, C2 = full_adder(A[-3], B[-3], C1)
    S3, Cout = full_adder(A[-4], B[-4], C2)
    S = [S3, S2, S1, S0]
    return Cout, S

print(adder4(A=[1, 0, 0, 1], B=[1, 1, 1, 0]))