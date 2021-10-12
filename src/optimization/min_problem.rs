use crate::{
    Result, DVec, DMat,
    optimization::{
        Region, WholeSpace,
        global_quadratic_minimizer, golden_search,
        ConstraintSet
    }
};



/// Data for problem    ? = argmin_{x in domain} objective_fn(x),
/// where domain is a convex region G.
///
/// Clearly we cannot expect to solve this problem on all convex regions,
/// even if the objective function is convex.
/// Typically this will only work if the location of the global minimum
/// is in G and the objective function f increases as we approach the
/// boundary of G.
/// This condition os satisfied by the barrier function on the feasible set
/// of a convex minimization problem and that is the principal application.
/// It may also work in some other cases, but there are no guarantees.
///
pub trait MinProblem {
    fn id(&self) -> String;
    fn dim(&self) -> usize;
    fn start_point(&self) -> DVec;
    fn objective_fn(&self, x: &DVec) -> f64;
    fn gradient(&self, x: &DVec) -> DVec;
    fn hessian(&self, x: &DVec) -> DMat;
    /// domain on which the objective function is minimized
    fn domain(&self) -> &dyn Region;

    /// determined by the behaviour of the objective function f along the line to
    /// the global minimizer of the quadratic approximation of f
    fn trust_radius(&self) -> f64 {
        let x = self.start_point();
        let g = self.gradient(&x);
        let H = self.hessian(&x);
        // small lambda puts glm far out
        let lambda = 0.001f64.min(self.gradient(&x).norm() / 10f64);
        match global_quadratic_minimizer(&x, &g, &H, lambda) {
            Err(e) => 1.0,
            Ok(glm) => {
                let D = self.domain();
                let glm_G = D.retract(&x, &glm);
                let d = &glm_G - &x;

                // line search in direction of glm
                let phi = |t: f64| self.objective_fn(&(&x + t * &d));
                let ls_result = golden_search(&phi, 0f64, 1f64, 0.1);
                let t_ls = ls_result.0;
                (t_ls * &d.norm()).max(1.0)
            }
        }
    }
}


/// Given a convex problem ? = argmin f(x) subject to g_i(x) <= 0 this is the (global)
/// MinProblem at the outer iteration of the barrier method where we minimize the function
///     h(x) = t*f(x) + barrierPenalty(x)
/// with barrierPenalty(x) being the barrier penalty of the constraint set.
/// This does not allow equality constraints.
/// The parameter t controls the duality gap.
///
trait BarrierSubProblem {

    fn id(&self) -> String;
    fn dim(&self) -> usize;
    fn t(&self) -> f64;
    fn start_point(&self) -> DVec;
    fn objectiveFn(&self,x: &DVec) -> f64;
    fn objectiveGradient(&self,x: &DVec) -> DVec;
    fn objectiveHessian(&self,x: &DVec) -> DMat;
    fn barrierFn(&self,x: &DVec) -> f64;
    fn barrierGradient(&self,x: &DVec) -> DVec;
    fn barrierHessian(&self,x: &DVec) -> DMat;
    fn domain(&self) -> &dyn Region;
}
impl MinProblem for BarrierSubProblem {

    fn id(&self) -> String {
        self.id()+" at t = "+self.t().toString().as_str()
    }
    fn dim(&self) -> usize { self.dim() }
    fn start_point(&self) -> DVec { self.start_point() }
    fn objective_fn(&self, x: &DVec) -> f64 {
        self.t()*self.objectiveFn(x) + self.barrierFn(x)
    }
    fn gradient(&self, x: &DVec) -> DVec {
        self.t()*self.objectiveGradient(x) + self.barrierGradient(x)
    }
    fn hessian(&self, x: &DVec) -> DMat {
        self.t()*self.objectiveHessian(x) + self.barrierHessian(x)
    }
    /// domain on which the objective function is minimized
    fn domain(&self) -> &dyn Region { self.domain() }
}




/// Barrier subproblem to determine feasibility of a set of inequality constraints
/// (no equality constraints allowed).
///
/// This is the subproblem in the outer iteration of the barrier method for minimization of the
///     objective function f(x,r) := r
///     subject to the constraints g(x) <= r,
/// for all constraints g(x) <= 0 in the constraintSet via the barrier method.
///
pub struct FeasibilitySubProblem {
    pub id: String,
    pub dim: usize,
    pub t: f64,
    pub region: WholeSpace,
    /// Constraintset for the constraints g(x)-r <= 0, not the original g(x) <= 0!
    pub constraintSet: ConstraintSet
}
impl FeasibilitySubProblem {
    pub fn new(t: f64, constraintSet: &ConstraintSet) -> FeasibilitySubProblem {

        let dim: usize = 1+constraintSet.dim;
        let id = String::from("Feasibility problem  for constraint set ") +
            constraintSet.id;
        FeasibilitySubProblem {
            id,dim,t,region:WholeSpace::new(dim),
            constraintSet: constraintSet.feasibility_constraint_set()
        }
    }
}
impl BarrierSubProblem for FeasibilitySubProblem {

    fn id(&self) -> String { self.id.clone() }
    fn dim(&self) -> usize { self.dim }
    fn t(&self) -> f64 { self.t }
    fn start_point(&self) -> DVec {

        let zeros: DVec = DVec::repeat(self.dim-1,0f64);
        let maxVal: f64 = self.constraintSet.iter().map(|ct| ct.value(zeros)).max();
        DVec::from_fn(self.dim,|i,_| if i<self.dim { 0f64 } else { 1f64+maxVal })
    }
    fn objectiveFn(&self, x: &DVec) -> f64 { x[self.dim-1] }
    fn objectiveGradient(&self, x: &DVec) -> DVec {
        DVec::from_fn(self.dim, |i,_| if i<self.dim { 0f64 } else { 1f64 })
    }
    fn objectiveHessian(&self, x: &DVec) -> DMat { DMat::zero(self.dim,self.dim) }
    fn barrierFn(&self, x: &DVec) -> f64 { self.constraintSet.log_barrier_value(x) }
    fn barrierGradient(&self, x: &DVec) -> DVec { self.constraintSet.log_barrier_gradient(x) }
    fn barrierHessian(&self, x: &DVec) -> DMat { self.constraintSet.log_barrier_hessian(x) }
    /// this problem is solved on the entire space
    fn domain(&self) -> &dyn Region { &WholeSpace{ dim: self.dim } }

}



/// BarrierSubProblem with duality gap parameter t for the feasibility problem of the
/// constraint set.
pub fn feasibility_sub_problem(t: f64, constraintSet: &ConstraintSet) -> FeasibilitySubProblem {

    FeasibilitySubProblem::new(t,constraintSet)
}