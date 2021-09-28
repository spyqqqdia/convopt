
pub use self::newton::*;
pub use self::solve::*;
pub use self::linesearch::*;

use crate::{Result, DVec, DMat};


mod linesearch;
mod newton;
mod solve;


//--------------------- Domains -------------------//

/// Convex subset of R^n.
///
pub trait Region {

    fn id(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn contains(&self,x: &DVec) -> bool;
    /// Assuming that x_0 is in this region G, computes the point
    /// u in G on the line(x_0,x) closest to x (pulls back x along the line to x_0
    /// until G is hit). Yields x if x is in G.
    fn retract(&self,x_0:&DVec,x: &DVec) -> DVec {

        assert!(x_0.len()==self.dim() && x.len()==self.dim(),
                "Dimension mismatch: dim(Region {}) = {}, dim(x_0) = {} and dim(x) = {}",
                self.id(), self.dim(),x_0.len(),x.len()
        );
        assert!(self.contains(x_0),"x_0 not in this Region");

        if self.contains(x) { return x.clone(); }

        // now x is not in G = this region
        let mut a = 0f64;
        let mut b = 1f64;
        let d = x-x_0;

        // maintain invariants (b > a), (x_0+a*d in G) and (x_0+b*d not in G)
        while b-a > 1e-10 {

            let c = (a+b)/2f64;
            let z = x_0 + c*&d;
            if self.contains(&z) { a=c; } else { b=c; }
        }
        x_0 + a*&d
    }
}



/// Region: the first orthant, all x_i>0.
pub struct WholeSpace {
    pub dim: usize
}
impl WholeSpace {
    pub fn new(dim: usize) -> WholeSpace { WholeSpace{ dim } }
}
impl Region for WholeSpace {

    fn id(&self) -> &'static str { &"WholeSpace" }
    fn dim(&self) -> usize { self.dim }
    fn contains(&self,x: &DVec) -> bool { true }
}



/// Region: the first orthant, all x_i>0.
pub struct AllPositive {
    pub dim: usize
}
impl AllPositive {
    pub fn new(dim: usize) -> AllPositive { AllPositive{ dim } }
}
impl Region for AllPositive {

    fn id(&self) -> &'static str { &"AllPositive" }
    fn dim(&self) -> usize { self.dim }
    fn contains(&self,x: &DVec) -> bool { x.min()>0f64 }
}




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

    fn id(&self) -> &'static str;
    fn dim(&self) -> usize;
    fn start_point(&self) -> DVec;
    fn objective_fn(&self,x:&DVec) -> f64;
    fn gradient(&self,x:&DVec) -> DVec;
    fn hessian(&self,x:&DVec) -> DMat;
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
        match global_quadratic_minimizer(&x,&g,&H,lambda) {

            Err(e) => 1.0,
            Ok(glm) => {
                let D = self.domain();
                let glm_G = D.retract(&x,&glm);
                let d = &glm_G-&x;

                // line search in direction of glm
                let phi = |t:f64| self.objective_fn(&(&x+t*&d));
                let ls_result = golden_search(&phi,0f64,1f64,0.1);
                let t_ls = ls_result.0;
                (t_ls*&d.norm()).max(1.0)
            }
        }
    }
}