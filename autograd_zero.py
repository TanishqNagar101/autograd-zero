import graphviz as gv
# import weakref
import math

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
        return Neuron(ans,[self],"tanh")       
    def relu(self):
        ans = max(0,self.val)
        return Neuron(ans,[self],"relu")
    def sigmoid(self):
        out = math.exp(-self.val)
        ans = 1/(1+out)
        return Neuron(ans,[self],"sigmoid")
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
            pkg.parents[0] += 1*pkg.grad
            pkg.parents[1] += -1*pkg.grad

        pkg._backward = _back_prop
        return pkg 
    def __rsub__(self,other):
        other = Neuron._ensure_neuron(other)
        pkg = Neuron(other.val-self.val,[other,self],"-")

        def _back_prop():
            pkg.parents[0].grad += -1*pkg.grad
            pkg.parents[1].grad += 1*pkg.grad
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
    
    #Going to implement in version-2
    def __truediv__(self,other):
        pass
    def __floordiv__(self,other):
        pass
    def __rpow__(self,other):
        pass
    def __rtruediv__(self,other):
        pass
    def __rfloordiv__(self,other):
        pass

#Loss Functions
    def mse(self,other):
        pass


#Basic Utlity Function
    def __repr__(self):
        return f"{self.val}"

#BackPropogation
    def backward(self):
        ans=[]
        vis=[False for _ in range(Neuron._counter)]
        Neuron._toposort(self,vis,ans)
        self.grad = 1.0
        print(ans)
        ans = ans[::-1]
        for i in ans:
            print(i)
            i._backward()
        return ans
        
        
    def zero_grad(self):
        pass

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


a=Neuron(54)
b=26
c = a*b
d = c+10
Loss = 30-d
type(Loss)
Loss.backward()
Loss.drawGraph()
