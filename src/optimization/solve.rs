
use crate::{
    error::ConvOptError, error::ErrKind,
    Result, DVec, DMat, FUN_nD_TO_1D,
    logging::Logger
};

use super::newton::*;
use super::{MinProblem,Region};


/// The regularization parameter lambda in newton::newton_step
/// r: trust radius
/// g: gradient at current iterate
///
fn reg_lambda(r: f64, g: &DVec) -> f64 {

    ((0.05f64/r.sqrt())*g.norm().max(0.00000001)).min(0.01)
}


pub fn solve_min_problem(min_prob: &impl MinProblem, eps:f64, max_iter:usize) -> Result<DVec> {

    let mut iter = 0;
    let rho = eps*(min_prob.dim() as f64).sqrt();
    let mut logger = Logger::new(format!("results/{}.log",min_prob.id()).as_str());

    let mut x = min_prob.start_point();
    let mut r= min_prob.initial_trust_radius();
    let mut grad = min_prob.gradient(&x);
    let mut lambda = reg_lambda(r, &grad);

    logger.write(format!("\n\nOptimization starts at point {}",&x).as_str());
    logger.write(format!(
        "f(x): {0:.2}, ||grad(f)(x)||: {1:.3}",min_prob.objective_fn(&x),grad.norm()
    ).as_str());

    while(iter<=max_iter && grad.norm()>=rho){

        let step = newton_step(&x, min_prob, r, lambda)?;
        logger.write(format!("\n\nIteration: {}, step: {}",iter,&step).as_str());
        x = step.next_point;
        r = step.new_trust_radius;
        grad = min_prob.gradient(&x);
        lambda = reg_lambda(r, &grad);
        iter +=1;
    }
    if iter==max_iter && grad.norm()>=rho {
        Err(ConvOptError::new(ErrKind::ConvergenceFailure("Max iterations hit")))
    } else {
        Ok(x)
    }
}

