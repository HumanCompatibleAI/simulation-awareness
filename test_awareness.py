from __future__ import division
# import matplotlib
import numpy as np
from scipy import optimize as opt
import sympy as s
import matplotlib.pyplot as plt
import copy as copy

#%% plot for a given testing

def draw(alpha=1,x0=-1,y0=-1,k=1,xt=0,xm=0.5, fignum=None):
    ym=(1-xm**alpha)**(1/alpha)
    
    plt.figure(fignum)
    plt.clf()
    ax = plt.subplot()
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    X = np.linspace(0., 2.)
    Y = np.linspace(0., 2.)[:, None]
    plt.contour(X, Y.ravel(), X**alpha + Y**alpha, [1])
    plt.title('Test scenario parameters')
    
    ax.axvline(x=x0)
    ax.text(x0,0,r'$x_0='+str(x0)+'$',rotation=90, fontsize=20, ha='right')
    
    ax.axhline(y=x0, )
    ax.text(0,y0,r'$y_0='+str(x0)+'$',fontsize=20, va='bottom')
    
    a=np.asarray([[0,1,0]])
    im=plt.imshow(a,extent=[xt-1/k,xt+1/k,-2,2],origin='lower',alpha=0.5,aspect='auto',cmap='Greys')
    
    ax.scatter(xm,ym)
    lab=r'$(x,y)=('+str(round(xm,3))+','+str(round(ym,3))+')$'
    ax.text(xm,ym,lab,fontsize=10, va='bottom')
    plt.show()

# draw(alpha=2,fignum=0)

#%% function for plotting single-variable slices of an expression  
def slice_func(expression, variable, **constants):
    if not expression.free_symbols.issubset(s.symbols(set(constants))
                                            | {variable}):
        s1=str(expression.free_symbols)+' !<= \n'
        s2=str(s.symbols(set(constants))| {variable})+'\n'            
        raise ValueError('Expression symbols not a subset of given symbols:\n'
                         +s1+s2)
    constants.pop(str(variable),None)
    f = expression.subs(constants)
    return np.vectorize(s.lambdify(variable, f))

def slice_plot(expression, variable, domain, fignum=None, **constants):
    f = slice_func(expression, variable, **constants)
    plt.figure(fignum)
    plt.plot(domain, f(domain))
    plt.show()

#%% model expressions
alpha,k,x_0,y_0,x_t,x = s.symbols('alpha k x_0 y_0 x_t x')

p_on = 1/(1+s.exp(-k*(x-x_t)))
y = (1-x**alpha)**(1/alpha)
eu_m = s.simplify(p_on*y + (1-p_on)*y_0)
eu_h = s.simplify(p_on*x + (1-p_on)*x_0)

# print("expected utility of machine", eu_m)

#%% function for computing optimal human x_t

d_eu_m = s.simplify(s.diff(eu_m,x))       
# print("derivative of EU of machine wrt human value", d_eu_m)

def add_to_dict(kv_list, dictionary):
    new_dict = copy.copy(dictionary)
    for (key, value) in kv_list:
        new_dict[key] = value
    return new_dict

def neg_eu_m_func_x(values):
    ''' 
    accepts a dict of values for all variables other than x:
    alpha, k, x_0, y_0, x_t
    returns a function from x to the value of -eu_m
    '''
    return lambda x: (-1) * eu_m.subs(add_to_dict([('x', x)], values))

def neg_d_eu_m_func_x(values):
    ''' 
    accepts a dict of values for all variables other than x:
    alpha, k, x_0, y_0, x_t
    returns a function from x to the value of -d_eu_m
    '''
    return lambda x: (-1) * d_eu_m.subs(add_to_dict([('x', x)], values))

def opt_machine_val_x(values):
    ''' 
    accepts a dict of values for all variables other than x:
    alpha, k, x_0, y_0 and x_t
    returns the value of x delivered by the machine
    '''
    if values['alpha'] == 2:
        return opt.minimize_scalar(neg_eu_m_func_x(values), bounds=(0.0, 1.0),
                                   method='bounded').x
    else:
        return opt.minimize_scalar(neg_eu_m_func_x(values)).x

def neg_eu_h_func_x_t(values):
    '''
    accepts a dict of values for alpha, k, x_0, and y_0
    returns a function from x_t to the expected utility of the human following
    the policy specified by x_t and k
    '''
    def x_val(x_t):
        new_vals = add_to_dict([('x_t', x_t)], values)
        return opt_machine_val_x(new_vals)
    return lambda x_t: (-1)*eu_h.subs(add_to_dict([('x_t', x_t),
                                                   ('x', x_val(x_t))],
                                                  values))

def equilibrium_outcome(values):
    '''
    accepts a dict of values for the variables alpha, k, x_0, and y_0
    returns a dict of the value of x_t that induces the greatest expected
    utility, as well as the value of x that the machine outputs and the 
    probability that the human lets the machine loose
    '''
    if values['alpha'] == 2:
        opt_x_t = opt.minimize_scalar(neg_eu_h_func_x_t(values),
                                      bounds=(values['x_0'], 1.0),
                                      # bounds=(0.0, 1.0),
                                      method='bounded').x
    else:
        opt_x_t = opt.minimize_scalar(neg_eu_h_func_x_t(values)).x
    new_vals = add_to_dict([('x_t', opt_x_t)], values)
    x_val = opt_machine_val_x(new_vals)
    prob_on = p_on.subs(add_to_dict([('x', x_val)], new_vals))
    return {'x_t': opt_x_t, 'x': x_val, 'p_on': prob_on}

#%% optimization with slice_func
defaults = {'alpha':1, 'k':10, 'x_0':-1, 'y_0':-5, 'x_t':0.5, 'x':0.9}
# print(eu_m)

EU_m = slice_func(eu_m,x,**defaults)
# print(eu_m.subs(defaults))

#%%
# slice_plot(eu_m,x,np.linspace(-2,2),fignum=1,
#            **{'alpha':1, 'k':8, 'y_0':-5, 'x_t':1})

#%%
# slice_plot(eu_m,x,np.linspace(0,1),fignum=2,
#            **{'alpha': 2, 'k': 8, 'y_0': 0.5, 'x_t': 0.95})
#%%

# opt_machine_val_linear = opt_machine_val_x(defaults)
# print("optimal machine value in linear pareto frontier", opt_machine_val_linear)
# neg_eu_h_func = neg_eu_h_func_x_t(defaults)
# exp_util_defaults_linear = (-1)*neg_eu_h_func(defaults['x_t'])
# print("expected human utility in linear pareto frontier under default policy",
#       exp_util_defaults_linear)
opt_human_policy_linear = equilibrium_outcome(defaults)
print("optimal policy in linear pareto frontier", opt_human_policy_linear)

# opt_machine_val_quad = opt_machine_val_x({'alpha': 2, 'k': 8, 'y_0': 0.5,
#                                           'x_0': 0.1, 'x_t': 0.9})
# print("optimal machine policy in circle pareto frontier", opt_machine_val_quad)
opt_human_policy_quad = equilibrium_outcome({'alpha': 2, 'k': 20, 'y_0': 0.5,
                                       'x_0': 0.1})
print("optimal policy in circle pareto frontier", opt_human_policy_quad)

opt_human_policy_cub  = equilibrium_outcome({'alpha': 3, 'k': 20, 'y_0': 0.5,
                                       'x_0': 0.1})
print("optimal policy in cubic pareto frontier", opt_human_policy_cub)

#%%

def plot_equilibrium(params, variable, values):
    '''
    params: dict containing 3 of 'alpha', 'k', 'y_0', 'x_0' and their values
    variable: one of 'alpha', 'k', 'y_0', 'x_0' that isn't contained in params
    values: range over which variable varies
    returns plot of equilibrium outcomes with given params and varying values of
    variable
    '''
    assert variable in ['alpha', 'k', 'y_0', 'x_0'], "'variable' input to plot_equilibrium not a valid variable"
    x_t_array = []
    x_array = []
    dict_string = ''.join('%s: %f ' % (key, val) for key, val in params.items())
    for value in values:
        my_dict = add_to_dict([(variable, value)], params)
        outcome = equilibrium_outcome(my_dict)
        x_t_array.append(outcome['x_t'])
        x_array.append(outcome['x'])
    plt.plot(values, x_t_array, 'bo', label="x_t")
    plt.plot(values, x_array, 'o', color="#ffa500", label="x")
    plt.title('x_t and x as a function of %s, fixing other variables as\n %s'
              % (variable, dict_string))
    plt.xlabel(variable)
    plt.legend()
    plt.show()

plot_equilibrium({'alpha': 2, 'y_0': 0.5, 'x_0': 0.1}, 'k', range(50))
plot_equilibrium({'alpha': 2, 'y_0': 0.5, 'k': 20}, 'x_0', np.arange(0,1,0.025))
    
    
