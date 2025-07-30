import graphviz as gv
import math
import random

class Neuron:
    _counter = 0
    def __init__(self,val,parent=[],operation="",back = lambda:None):
        self.val = val
        self.parents = parent
        self.operation = operation
        self.grad = 0.0
        self._id = Neuron._counter
        self._backward = back
        Neuron._counter +=1

#Activation Function

    def tanh(self):
        """Tanh activation function"""
        ans = math.tanh(self.val)
        pkg = Neuron(ans,[self],"tanh") 

        def _back_prop():
            pkg.parents[0].grad += pkg.grad*(1-pkg.val**2)
        
        pkg._backward = _back_prop
        return pkg       
    def relu(self):
        ans = max(0,self.val)
        pkg = Neuron(ans,[self],"relu")

        def _back_prop():
            x = pkg.parents[0]
            if(x.val>0):
                x.grad +=pkg.grad
        pkg._backward = _back_prop 
        return pkg 
    def sigmoid(self):
        out = math.exp(-self.val)
        ans = 1/(1+out)
        pkg = Neuron(ans,[self],"sigmoid")

        def _back_prop():
            x = pkg.parents[0]
            x.grad += pkg.grad*pkg.val * (1-pkg.val)

        pkg._backward = _back_prop
        return 
    def lrelu(self):
        ans = self.val
        if(self.val< 0):
            ans = ans * 0.1
        return Neuron(ans,[self],"lrelu")
    def linear(self):
        return Neuron(self.val,[self],"linear")
    
    #Going to implement in version-2
    def softmax(self):
        pass
    def gelu(self):
        pass
    def selu(self):
        pass
    def maxout(self):
        pass 


#Arithmetic functions
    def __add__(self,other):
        other = Neuron._ensure_neuron(other)
        pkg = Neuron(self.val+other.val,[self,other],"+")
        
        def _back_prop():
            pkg.parents[0].grad += 1*pkg.grad
            pkg.parents[1].grad += 1*pkg.grad

        pkg._backward = _back_prop

        return pkg
    def __radd__(self,other):
        return self.__add__(other)
    def __mul__(self,other):
        other = Neuron._ensure_neuron(other)
        pkg = Neuron(self.val*other.val,[self,other],"*")
        
        def _back_prop(): 
            pkg.parents[0].grad += pkg.grad*pkg.parents[1].val
            pkg.parents[1].grad += pkg.grad*pkg.parents[0].val
        
        pkg._backward = _back_prop
        return pkg
    def __rmul__(self,other):
        return self.__mul__(other)
    def __sub__(self,other):
        other = Neuron._ensure_neuron(other)
        pkg = Neuron(self.val-other.val,[self,other],"-")

        def _back_prop():
            pkg.parents[0].grad += 1*pkg.grad
            pkg.parents[1].grad += -1*pkg.grad

        pkg._backward = _back_prop
        return pkg 
    def __rsub__(self,other):
        other = Neuron._ensure_neuron(other)
        pkg = Neuron(other.val-self.val,[other,self],"-")

        def _back_prop():
            pkg.parents[0].grad += 1*pkg.grad
            pkg.parents[1].grad += -1*pkg.grad
        pkg._backward = _back_prop
        return pkg
    def __pow__(self,other):
        other = Neuron._ensure_neuron(other)
        pkg = Neuron(self.val**other.val,[self,other],"power")
        
        def _back_prop():
            x = pkg.parents[0].val 
            if(x<=0):
                x+=1e-10 
            pkg.parents[0].grad += pkg.grad * (pkg.parents[1].val * (x**(pkg.parents[1].val-1)))
            pkg.parents[1].grad += pkg.grad *pkg.val * math.log(x)
        
        pkg._backward = _back_prop
        return pkg
    
    #Got to implement in version-2
    def __truediv__(self,other):
        other = Neuron._ensure_neuron(other)
        if(other.val==0):
            raise ZeroDivisionError("Division by zero is not allowed")
        
        pkg =Neuron(self.val/other.val,[self,other],"/")

        def _back_prop():
            x = pkg.parents[0]
            y = pkg.parents[1]
            x.grad += pkg.grad*(1/y.val)*1
            y.grad += pkg.grad*x.val*(-1/(y.val**2)) 
        
        pkg._backward = _back_prop
        return pkg 
    def __rpow__(self,other):
        other = Neuron._ensure_neuron(other)
        pkg = Neuron(self.val**other.val,[other,self],"power")
        
        def _back_prop():

            x = pkg.parents[0]
            y = pkg.parents[1]
            z = x.val
            if(z<=0):
                z+=1e-10 
            x.grad += pkg.grad * (y.val * z ** (y.val - 1))
            y.grad += pkg.grad * pkg.val * math.log(z)
        
        pkg._backward = _back_prop
        return pkg
    def __rtruediv__(self,other):
        other = Neuron._ensure_neuron(other)
        if(self.val==0):
            raise ZeroDivisionError("Division by zero is not allowed")
        
        pkg =Neuron(other.val/self.val,[other,self],"/")

        def _back_prop():
            x = pkg.parents[0]
            y = pkg.parents[1]
            x.grad += pkg.grad*(1/y.val)*1
            y.grad += pkg.grad*x.val*(-1/(y.val**2)) 
        
        pkg._backward = _back_prop
        return pkg 
    
    #Not Implementable
    def __rfloordiv__(self,other):
        """It is not diffrentable due to its jumpy behaviour like for 4.999999 it is 4 but for 5.00 it is 5"""
        raise Exception("Not Supported due to it's non differential behaviour")
    def __floordiv__(self,other):
        """It is not diffrentable due to its jumpy behaviour like for 4.999999 it is 4 but for 5.00 it is 5"""
        raise Exception("Not Supported due to it's non differential behaviour")


#Loss Functions
    def mse(self,other):
        other = Neuron._ensure_neuron(other)
        ans = 0.5*(other.val-self.val)
        pkg = Neuron(ans,[other,self],"Mean Square Error")

        def _back_prop():
            x = pkg.parents[0]
            y = pkg.parents[1]
            temp = 2*(x.val-y.val)
            x.grad += temp
            y.grad += -temp
        
        pkg._backward = _back_prop
        return pkg



#Basic Utlity Function
    def __repr__(self):
        return f"{self.val}"

#BackPropogation
    def backward(self):
        ans=[]
        vis=[False for _ in range(Neuron._counter)]
        Neuron._toposort(self,vis,ans)
        self.grad = 1.0
        ans = ans[::-1]
        for i in ans:
            i._backward()
        return ans
        
        
    def zero_grad(self):
        nodes = []
        vis = [False for _ in range(Neuron._counter)]
        Neuron._toposort(self, vis, nodes)
        for node in nodes:
            node.grad = 0.0
#visual functions
    def drawGraph(self):
        dot = gv.Digraph(comment=f"x")
        dot.attr(rankdir="LR")
        vis = set()
        c = [0]
        Neuron._dfs(self,vis,dot,c)
        dot.render("doctest-output/x.gv",view=True)
        return dot


#Helper Functions
    def get(self):
        return{"Value":self.val,
               "Parents":[p() for p in self.parents],
               "Operation":self.operation,
               "Gradient":self.grad}
    
    @staticmethod
    def _ensure_neuron(other):
        """Ensure during the operation that all elements are of Neuron Class"""
        if(isinstance(other,Neuron) ):
            return other
        elif(isinstance(other,(int,float))):
            return Neuron(other)
        else:
            raise Exception(f"Not support for operand type:- {type(other)}")
    
    @staticmethod
    def _toposort(neuron,vis,ans):
        vis[neuron._id] = True
        for i in neuron.parents:
            if(not vis[i._id]):
                Neuron._toposort(i,vis,ans)
        ans.append(neuron)


    @staticmethod
    def _dfs(neuron,vis,dot,idx):
        """This need to be optimized using Topological Sort and naming convention needs to be changed"""

        if(neuron in vis):
            return
        dot.node(f"n_{neuron._id}",f"{{ {neuron.val} | {neuron.grad} }}",shape="record")
        vis.add(neuron)
        if(neuron.operation == ""):
            return
        dot.node(f"op_{idx[0]}",neuron.operation,shape ="oval")
        dot.edge(f"op_{idx[0]}",f"n_{neuron._id}")
        for p in neuron.parents:
            if(p == None or p in vis):
                continue
            dot.node(f"n_{p._id}",f"{{{p.val}|{p.grad}}}", shape="record")
            dot.edge(f"n_{p._id}",f"op_{idx[0]}")
        idx[0]+=1
        
        for p in neuron.parents:
            if(p == None):
                continue
            Neuron._dfs(p,vis,dot,idx)


class singleNeuron:
    def __init__(self,nin):
        self.w = [Neuron(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Neuron(random.uniform(-1,1))
    
    def __call__(self,x):
        ans = sum((wi*xi for wi,xi in zip(self.w,x)),self.b)
        out = ans.tanh()
        return out
    
class Layer:
    def __init__(self,nin,nout):
        self.neurons = [singleNeuron(nin) for _ in range(nout)]

    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

class MLP:
    def __init__(self,nin,nout):
        sz = [nin]+nout
        self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nout))]
    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    a = [2,3,4]
    n = MLP(3,[4,4,1])
    y = 15
    y_pred = n(a)
    loss = y_pred.mse(y)
    # loss.drawGraph()
    loss.backward()
    loss.drawGraph()
