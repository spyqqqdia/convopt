use crate::{
    error::ConvOptError, error::ErrKind,
    Result, DVec, DMat, FUN_nD_TO_1D,
    logging::Logger,
    optimization::{Region, WholeSpace, AllPositive, MinProblem}
};



/// Minimize the negentropy (i.e. maximize the entropy) of a discrete distribution
/// with n outcomes (atoms).
/// The objective function is f(p) = \sum_{i=1}^n p_i*log(p_i) + 10*(1-\sum_ip_i)^2
/// on the domain p>0.
/// The penalty term 10*(1-\sum_ip_i)^2 is a replacement for the constraint \sum_ip_i=1.
/// The minimum is attained near p_i=1/n with value near -log(n), all coordinates must be equal
/// due to symmetry of the objective function under variable permutation.
///
pub struct Maxent {
    dim: usize,
    G: AllPositive,
}
impl Maxent {
    pub fn new(n:usize) -> Maxent { Maxent { dim: n, G: AllPositive::new(n) } }
}
impl MinProblem for Maxent {

    fn id(&self) -> &'static str { "Maxentproblem" }
    fn dim(&self) -> usize { self.dim }
    fn start_point(&self) -> DVec {
        DVec::from_fn(self.dim,|i,_| if i==0 { 0.9 } else { 0.1 })
    }
    fn objective_fn(&self,x: &DVec) -> f64 {
        let q = 1f64 - x.sum();
        x.map(|u| u*u.ln()).sum() + 10f64*q*q
    }
    fn gradient(&self,x: &DVec) -> DVec {
        let s = 1f64-x.sum();
        DVec::from_fn(self.dim,|i,_| 1f64+x[i].ln()-20f64*s)
    }
    fn hessian(&self,x: &DVec) -> DMat {
        DMat::from_fn(self.dim,self.dim,
            |i,j| if i==j { 1f64/x[i]+20f64 } else { 20f64 }
        )
    }
    fn domain(&self) -> &dyn Region { &(self.G) }
}



/// The Rosenbrook function f(x,y) = (x-a)² + b*(y+x²)². Minimum at (a,-a²).
/// This is a convex variation of the Rosenbrook function which has the term b*(y-x²)² instead.
///
pub struct Rosenbrook {
    dim: usize,
    G: WholeSpace,
    a: f64,
    b: f64,
}
impl Rosenbrook {
    pub fn new(a:f64,b:f64) -> Rosenbrook {
        Rosenbrook { dim: 2, G: WholeSpace::new(2), a, b }
    }
}
impl MinProblem for Rosenbrook {

    fn id(&self) -> &'static str { "ArgminRosenbrook" }
    fn dim(&self) -> usize { self.dim }
    fn start_point(&self) -> DVec {
        DVec::from_fn(self.dim,|i,_| if i==0 { 6f64 } else { 2f64 })
    }
    fn objective_fn(&self,x: &DVec) -> f64 {
        let q = x[0]-self.a;
        let r = x[1]+x[0]*x[0];
        q*q + self.b*r*r
    }
    fn gradient(&self,x: &DVec) -> DVec {
        let r = x[1]+x[0]*x[0];
        let f_x = 2f64*(x[0]-self.a) + 4f64*self.b*r*x[0];
        let f_y = 2f64*self.b*r;
        DVec::from_row_slice(&[f_x,f_y])
    }
    fn hessian(&self,x: &DVec) -> DMat {

        let f_xx = 2f64 + 4f64*self.b*x[1]+12f64*self.b*x[0]*x[0];
        let f_xy = 4f64*self.b*x[0];
        let f_yy = 2f64*self.b;
        DMat::from_row_slice(2,2,&[
            f_xx, f_xy,
            f_xy, f_yy
        ])
    }
    fn domain(&self) -> &dyn Region { &(self.G) }
}