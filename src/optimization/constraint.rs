use std::fmt;
use crate::{
    error::ConvOptError, error::ErrKind,
    equation::cholesky_solve_regularized,
    Result, DVec, DMat, FUN_nD_TO_1D,
    logging::Logger,
    optimization::{MinProblem, golden_search},
    matrix_utils::cross_product
};

use super::Region;
use std::cmp::{min, max};

//------------------ Inequality constraints ------------------//
//
// WARNING: for a constraint g(x)<=0 we are using the log-barrier -log(-g(x)).
// This will be NaN in case g(x)=0. Hence the optimization must stay in the region
// g(x) < 0.

/// Constraint g(x) <= 0, with g assumed to be convex, C2 and
/// defined everywhere.
///
pub trait InequalityConstraint: Clone {

    fn id(&self) -> String;
    fn dim(&self) -> usize;
    fn value(&self,x: &DVec) -> f64;
    fn gradient(&self,x: &DVec) -> DVec;
    fn hessian(&self,x: &DVec) -> DMat;
    /// -log(-g) for the constraint g(x) <= 0.
    /// Needed for log-barrier penalty function
    fn log_barrier_value(&self,x: &DVec) -> f64 { -(-self.value(x)).ln() }

    /// grad(-log(-g)) for the constraint g(x) <= 0.
    /// Needed for log-barrier penalty function
    fn log_barrier_gradient(&self,x: &DVec) -> DVec {

        let r = 1f64/self.value(x);
        r*self.gradient(x)
    }
    /// hessian(-log(-g)) for the constraint g(x) <= 0.
    /// Needed for log-barrier penalty function
    fn log_barrier_hessian(&self,x: &DVec) -> DMat {

        let r = 1f64/self.value(x);
        let g = self.gradient(x);
        r*(self.hessian(x)-r*cross_product(&g,&g))
    }
}


fn feasibility_constraint(ct: &impl InequalityConstraint) -> Box<dyn InequalityConstraint>
{
    #[derive(Clone,Debug)]
    struct result { ct: Box<dyn InequalityConstraint> };
    impl InequalityConstraint for result {

        fn id(&self) -> String { self.ct.id().clone()+" feasibility" }
        fn dim(&self) -> usize { 1+self.ct.dim() }
        fn value(&self,x: &DVec) -> f64 {

            let z = DVec::from_fn( self.ct.dim(),|i,_| x[i]);
            self.ct.value(&z)-x[self.ct.dim()]
        }
        fn gradient(&self,x: &DVec) -> DVec  {

            let z = DVec::from_fn( self.ct.dim(),|i,_| x[i]);
            self.ct.gradient(&z)   // FIX ME
        }
        fn hessian(&self,x: &DVec) -> DMat  {

            let z = DVec::from_fn( self.ct.dim(),|i,_| x[i]);
            self.ct.hessian(&z)   // FIX ME
        }
    }
    let res = result{ ct: Box::new(ct.clone()) };
    Box::new(res)
}


impl Region for dyn InequalityConstraint {

    fn id(&self) -> String {
        String::from("Feasibility region for ")+self.id().as_str()
    }
    fn dim(&self) -> usize {
        self.dim()
    }
    fn contains(&self,x: &DVec) -> bool {
        self.value(x) <= 0f64
    }
}

/// Constraint a'x <= c
#[derive(Clone,Debug)]
pub struct LinearInequalityConstraint {

    pub id: String,
    pub a: DVec,
    pub c: f64,
}

impl LinearInequalityConstraint {

    pub fn new(id: String, a: DVec, c: f64) -> LinearInequalityConstraint {
        LinearInequalityConstraint{ id,a,c}
    }
}

impl InequalityConstraint for LinearInequalityConstraint {

    fn id(&self) -> String { self.id.clone() }
    fn dim(&self) -> usize { self.a.len() }
    fn value(&self,x: &DVec) -> f64 { self.a.dot(x)-self.c }
    fn gradient(&self,x: &DVec) -> DVec { self.a.clone() }
    fn hessian(&self,x: &DVec) -> DMat {
        DMat::from_element(self.dim(),self.dim(),0f64)
    }
}



/// Constraint a'x+(1/2)x'Qx <= c
#[derive(Clone,Debug)]
pub struct QuadraticInequalityConstraint {

    pub id: String,
    pub a: DVec,
    pub Q: DMat,
    pub c: f64,
}

impl QuadraticInequalityConstraint {

    pub fn new(id: String, a: DVec, Q: DMat, c: f64) -> QuadraticInequalityConstraint {
        QuadraticInequalityConstraint{ id,a,Q,c}
    }
}

impl InequalityConstraint for QuadraticInequalityConstraint {

    fn id(&self) -> String { self.id.clone() }
    fn dim(&self) -> usize { self.a.len() }
    fn value(&self,x: &DVec) -> f64 {
        self.a.dot(x)+0.5f64*(&self.Q*x).dot(x)-self.c
    }
    fn gradient(&self,x: &DVec) -> DVec {
        self.a.clone()+(&self.Q*x)
    }
    fn hessian(&self,x: &DVec) -> DMat { self.Q.clone() }
}


/// Set of inequality constraints (equality constraints are handled separately).
///
pub struct ConstraintSet {

    pub id: String,
    pub dim: usize,
    pub constraints: Vec<Box<dyn InequalityConstraint>>,
}

impl ConstraintSet {

    pub fn new(id: String, dim: usize) -> ConstraintSet {

        ConstraintSet{ id, dim, constraints: Vec::new() }
    }
    pub fn add_constraint(&mut self,constraint: Box<dyn InequalityConstraint>) -> () {
        assert!(constraint.dim()==self.dim);
        self.constraints.push(constraint);
    }
    pub fn add_constraints(&mut self,constraints: Vec<Box<dyn InequalityConstraint>>) -> () {

        for ct in constraints { self.add_constraint(ct) } ;
    }
    /// Sum of -log(-f) over all constraints f(x)<=0.
    /// Needed for log-barrier penalty function
    pub fn log_barrier_value(&self,x: &DVec) -> f64 {
        self.constraints.iter().
            map(|ct: &Box<dyn InequalityConstraint>| -> f64 { ct.value(x) }).sum()
    }

    /// Sum of grad(-log(-f)) over all constraints f(x)<=0.
    /// Needed for log-barrier penalty function
    pub fn log_barrier_gradient(&self,x: &DVec) -> DVec {

        self.constraints.iter().
            map(|ct: &Box<dyn InequalityConstraint>| -> DVec { ct.log_barrier_gradient(x) }).sum()
    }
    /// Sum of hessian(-log(-f)) over all constraints f(x)<=0.
    /// Needed for log-barrier penalty function
    pub fn log_barrier_hessian(&self,x: &DVec) -> DMat {

        self.constraints.iter().
            map(|ct: &Box<dyn InequalityConstraint>| -> DMat { ct.log_barrier_hessian(x) }).sum()
    }
    /// the set of feasibility constraints g(x)-r <= 0 for all constraints g(x) <= 0
    /// in this constraint set.
    /// Needed for phase I feasibility analysis
    pub fn feasibility_constraint_set(&self) -> ConstraintSet {

        let cts: &Vec<Box<dyn InequalityConstraint>> = &self.constraints;
        let feasibility_constraints: Vec<Box<dyn InequalityConstraint>> =
            cts.iter().
                map(|ct: &Box<dyn InequalityConstraint>| -> Box<dyn InequalityConstraint> {
                    feasibility_constraint(ct)
            }).collect();
        let mut res = ConstraintSet::new(
            String::from("FeasibilityConstraintSet for ")+self.id.as_str(),
             1+self.dim
        );
        res.add_constraints(feasibility_constraints);
        res
    }
}