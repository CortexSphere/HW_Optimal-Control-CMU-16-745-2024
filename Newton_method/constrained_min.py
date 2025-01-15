import numpy as np
import casadi as ca

def objective_function(x):
    return 0.5*(pow(x[0]-1,2)/2.0+x[1]**2)
def constrained_funtion(x):
    return x[0]**2+2*x[0]-x[1]

class EqConstrainedNewtonMethod:
    def __init__(self,object,constrained,n_x,n_constraints,x0,lambda0,beta_init=1e-3,alpha_init=1,maxiter=100,tol=1e-3):
        self.object_func = object
        self.constrained_func = constrained
        self.n_x = n_x
        self.n_constraints = n_constraints
        self.beta_init = beta_init
        self.alpha_init = alpha_init
        self.x_sys=ca.MX.sym('x_sys', self.n_x)
        self.lambda_sys=ca.MX.sym('lambda', self.n_constraints)
        self.x=x0
        self.lambda_=lambda0
        self.maxiter=maxiter
        self.tol=tol
        self.get_Jac_Hessian()
    def get_Jac_Hessian(self):
        self.f=self.object_func(self.x_sys)
        self.fx=ca.jacobian(self.f, self.x_sys)
        self.fxx=ca.jacobian(self.fx, self.x_sys)
        self.fx_func=ca.Function("fx", [self.x_sys], [self.fx])
        self.fxx_func=ca.Function("fxx", [self.x_sys], [self.fxx])
        #Lagrange function
        self.L= self.object_func(self.x_sys) + self.lambda_sys.T * self.constrained_func(self.x_sys)
        #Jocobian and Hessian
        self.Lx=ca.jacobian(self.L, self.x_sys)
        self.Llmd=ca.jacobian(self.L, self.lambda_sys)
        self.Lxx=ca.jacobian(self.Lx, self.x_sys)
        self.Llmdlmd=ca.jacobian(self.Llmd, self.lambda_sys)
        self.Lxlmd=ca.jacobian(self.Lx, self.lambda_sys)
        self.Llmdx=ca.jacobian(self.Llmd, self.x_sys)

        self.L_func=ca.Function("L", [self.x_sys, self.lambda_sys], [self.L])
        self.Lx_func=ca.Function("Lx", [self.x_sys, self.lambda_sys], [self.Lx])
        self.Llmd_func=ca.Function("Llmd", [self.x_sys, self.lambda_sys], [self.Llmd])
        self.Lxx_func=ca.Function("Lxx", [self.x_sys, self.lambda_sys], [self.Lxx])
        self.Llmdlmd_func=ca.Function("Llmdlmd", [self.x_sys, self.lambda_sys], [self.Llmdlmd])
        self.Lxlmd_func=ca.Function("Lxlmd", [self.x_sys, self.lambda_sys], [self.Lxlmd])
        self.Llmdx_func=ca.Function("Llmdx", [self.x_sys, self.lambda_sys], [self.Llmdx])
    def is_positive_definite(self,Matrix):
        try:
            np.linalg.cholesky(Matrix)
            return True
        except np.linalg.LinAlgError:
            return False
    def solve(self):
        iter=0
        while iter < self.maxiter:
            iter=iter+1
            print("newton iteration:",iter)
            Lxx=self.Lxx_func(self.x, self.lambda_)
            Lxlmd=self.Lxlmd_func(self.x, self.lambda_)
            Llmdx=self.Llmdx_func(self.x, self.lambda_)
            Lx=self.Lx_func(self.x, self.lambda_)
            Llmd=self.Llmd_func(self.x, self.lambda_)
            fxx=self.fxx_func(self.x)
            # Full-Newton method

            beta=self.beta_init
            while not self.is_positive_definite(Lxx):
                Lxx+=beta*ca.DM.eye(self.n_x)
                beta*=2
            kkt_matrix=ca.vertcat(
                ca.horzcat(Lxx,Lxlmd),
                ca.horzcat(Llmdx,ca.DM.zeros((self.n_constraints,self.n_constraints)))
            )
            """
            # Gussian-Newton method
            #not necessary in GNM
            beta=self.beta_init
            while not self.is_positive_definite(fxx):
                fxx+=beta*ca.DM.eye(self.n_x)
                beta*=2
            kkt_matrix=ca.vertcat(
             ca.horzcat(fxx,Lxlmd),
             ca.horzcat(Llmdx,ca.DM.zeros((self.n_constraints,self.n_constraints)))
             )
             """
            rms = -ca.vertcat(
                Lx.T,Llmd
            )
            z=ca.solve(kkt_matrix, rms)
            delta_x=z[:self.n_x]
            delta_lmd=z[self.n_x:self.n_x+self.n_constraints]
            alpha=self.alpha_init
            c=1e-3
            while(self.L_func(self.x+alpha*delta_x,self.lambda_)>self.L_func(self.x,self.lambda_)+alpha*c*self.fx_func(self.x)@delta_x):
                alpha*=0.5
            #while(self.object_func(self.x+alpha*delta_x)>self.object_func(self.x)+alpha*c*self.fx_func(self.x)@delta_x):
                #alpha*=0.5
            self.x=self.x+alpha*delta_x
            self.lambda_=self.lambda_+alpha*delta_lmd
            print("x:",self.x,"lambda:",self.lambda_)
            newton_minus=pow(delta_x.T@self.fxx_func(self.x)@delta_x,0.5)
            print("newton_minus:",newton_minus)
            if newton_minus<self.tol:
                print("converged at iteration %d"%iter)
                break
        print("Maximum iterations reached")
        return self.x, self.lambda_

solver=EqConstrainedNewtonMethod(objective_function,constrained_funtion,2,1,ca.DM([-3,2]),ca.DM.zeros(1))

x_opt,lamda_opt=solver.solve()
print("Optimal x:", x_opt)
print("Optimal lambda:", lamda_opt)




















