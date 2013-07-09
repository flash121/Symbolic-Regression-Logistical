'''
Created on 2013-7-2

@author: yfeng
'''
#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
from sklearn import metrics
import math
import random
from datatest import getdata
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
co=0
def tryaces(input):
    return 1-input

def access_right(in1,in2):
    return in1

def access_left(in1,in2):
    return in2


def staticLimitCrossover(ind1, ind2, heightLimit): 
    # Store a backup of the original individuals 
    keepInd1, keepInd2 = toolbox.clone(ind1), toolbox.clone(ind2) 

    # Mate the two individuals 
    # The crossover is done in place (see the documentation) 
    # If using STGP (like spambase), replace this line by 
    gp.cxOnePoint(ind1, ind2)
    # If a child is higher than the maximum allowed, then 
    # it is replaced by one of its parent 
    if ind1.height > heightLimit: 
        ind1[:] = keepInd1 
    if ind2.height > heightLimit: 
        ind2[:] = keepInd2 
    return ind1,ind2

def staticLimitMutation(individual, expr, heightLimit): 
    # Store a backup of the original individual 
    keepInd = toolbox.clone(individual) 

    # Mutate the individual 
    # The mutation is done in place (see the documentation) 
    # If using STGP (like spambase), replace this line by 
    #gp.mutUniform(individual,expr) 
    gp.mutUniform(individual, expr)  
    # If the mutation sets the individual higher than the maximum allowed, 
    # replaced it by the original individual 
    if individual.height > heightLimit: 
        individual[:] = keepInd 
    return individual









pset =gp.PrimitiveSet("MAIN", 8, 'IN')
pset.addPrimitive(max,2)
pset.addPrimitive(min, 2)
pset.addPrimitive(tryaces, 1)
pset.addPrimitive(access_right, 2)
pset.addPrimitive(access_left, 2)
#pset.addEphemeralConstant(lambda: random.randint(-1,1))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genRamped, pset=pset, min_=0, max_=1)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("lambdify", gp.lambdify, pset=pset)
A,X=getdata()
#toolbox.decorate('bloat',de)
def evalSymbReg(individual):
    # Transform the tree expression in a callable function
    func = toolbox.lambdify(expr=individual)
    # Evaluate the sum of squared difference between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    #X=[[1.0000,0.9231,1.0000,1.0000,1.0000,1.0000,0.9091,1.0000]]
    #X.append([1.0000,0.9231,1.0000,1.0000,0.8333,0.5000,0.9091,0.8333])
    #X.append([1.0000,0.9231,1.0000,1.0000,0.8333,1.0000,0.9091,0.8333])
    #X.append([0.0000,0.9231,0.0000,0.0000,0.8333,0.5000,0.9091,0.8333])
    #X=[[1.0000,0,1.0000,1.0000,1.0000,1.0000,0,1.0000]]
    #X.append([1.0000,0,1.0000,1.0000,0,0.5000,0,0])
    #X.append([1.0000,0,1.0000,1.0000,0,1.0000,0,0])
    #X.append([1.0000,1,1.0000,1.0000,0,0.5000,0,1])
    #X.append([0.0000,1,1.0000,1.0000,0,0.5000,0,1])
    #L
    #A=[1,1,1,0,1];
    t=0
    a=0
    global co 
    co+=1
   # print co
    if(co>6000):
        global A,X
        A,X=getdata()
    preds=[]
    for x in X:
        preds.append(func(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7]))
        #t+=1
    auc = 1-metrics.auc_score(A, preds)
    return auc,

toolbox.register("evaluate", evalSymbReg)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", staticLimitCrossover, heightLimit=17)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register('mutate',  gp.mutUniform, expr=toolbox.expr_mut)
#toolbox.decorate("select",gp.staticDepthLimit(5))
#toolbox.decorate("mutate",gp.staticSizeLimit(100))

def main():
    random.seed(318)

    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", tools.mean)
    stats.register("std", tools.std)
    stats.register("min", min)
    stats.register("max", max)
    
    algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats, halloffame=hof)
    
    print hof

if __name__ == "__main__":
    main()