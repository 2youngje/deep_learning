class LogicGate:
    def __init__(self,w1,w2,bias):
        self.w1 = w1
        self.w2 = w2
        self.bias = bias

    def __call__(self,x1,x2) :
        if (x1 * self.w1) + (x2 * self.w2) + self.bias > 0 :
            y = 1
        else :
            y = 0
        return y

class ANDGate :
    def __init__(self):
        self.gate = LogicGate(w1=0.5, w2=0.5, bias=-0.7)

    def __call__(self,x1,x2):
        return self.gate(x1,x2)

class ORGate :
    def __init__(self):
        self.gate = LogicGate(w1=0.5, w2=0.5, bias=-0.2)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)

class NANDGate :
    def __init__(self):
        self.gate = LogicGate(w1=-0.5, w2=-0.5, bias=0.7)

    def __call__(self,x1,x2):
        return self.gate(x1,x2)

class NORGate :
    def __init__(self):
        self.gate = LogicGate(w1=-0.5, w2=-0.5, bias=0.2)

    def __call__(self, x1, x2):
        return self.gate(x1, x2)
class XORGate :
    def __init__(self):
        self.gate_and = ANDGate()
        self.gate_or = ORGate()
        self.gate_nand = NANDGate()

    def __call__(self,x1,x2):
        p = self.gate_nand(x1,x2)
        q = self.gate_or(x1,x2)
        z = self.gate_and(p,q)

        return z

and_gate = ANDGate()
or_gate = ORGate()
nand_gate = NANDGate()
xor_gate = XORGate()

print(and_gate(1,0))
print(or_gate(1,1))
print(nand_gate(0,1))
print(xor_gate(1,0))