use std::fmt;
use crate::{
    error::ConvOptError, error::ErrKind,
    equation::cholesky_solve_regularized,
    Result, DVec, DMat, FUN_nD_TO_1D,
    logging::Logger,
    optimization::{MinProblem, golden_search}
};

use super::Region;
use nalgebra::Vector;
use std::cmp::{min, max};


/// Solve the regularized Newton equation $(H+l*I)p=-g$ by Cholesky factorization of
/// $H+l*I$. The matrix $H+l*I$ needs to be positive definite, i.e. if $H$ is singular
/// we must have $l>0$.
///
/// # Arguments
///
/// * `H`: positive semidefinite symmetric matrix
/// * `l`: nonnegative scalar (regularization parameter)
/// * `g`: vector of same dimension as H
///
pub fn solve_newton_equation(g: &DVec, H: &DMat, l: f64) -> Result<DVec> {

    cholesky_solve_regularized(H,&(-g),l)
}





/// Structure containing all the relevant information for a Newton step from
/// the current iterate `current_point` to the next iterate `next_point`.
///
/// The Newton step will either move to the Cauchy point ("CP"), the Dog-leg point ("DLP")
/// the global minimizer p ("GM", equation Hp=-g) , the regularized global
/// minimizer p ("RGM", equation: (H+lI)p=-g) or the boundary minimizer p ("BM", minimizer on
/// ||p||<=r, equation (H+lI)p=-g with iteration on l to get ||p||=r) depending on which of these
/// points are computed and yield the most decrease in the objective function f.
///
/// All Newton step methods compute the Cauchy point CP but then it varies which other points
/// are computed also as candidates for the next iterate.
///
///
#[derive(Debug)]
pub struct NewtonStep {
    /// "CP", "DLP" or "GM" according as the Cauchy point "CP", Dog-leg point ("DLP")
    /// or global minimizer ("GM") of the quadratic approximation provided the most
    /// decrease in the objective function
    pub next_point_ID: &'static str,
    pub old_trust_radius: f64,
    pub new_trust_radius: f64,
    /// step size to line search point
    pub r_ls: f64,
    /// step size to cauchy point
    pub r_cp: f64,
    /// step size to dog leg point
    pub r_dlp: f64,
    /// step size to global quadratic optimizer
    pub r_glm: f64,
    /// decrease in value of objective function as %(f(current iterate)) at  line search point
    pub ls_decrease: f64,
    /// decrease in value of objective function as %(f(current iterate)) at Cauchy point
    pub cp_decrease: f64,
    /// decrease in value of objective function as %(f(current iterate)) at dog leg point
    pub dlp_decrease: f64,
    /// decrease in value of objective function as %(f(current iterate)) at global quadratic
    /// minimizer
    pub glm_decrease: f64,
    /// current iterate
    pub current_point: DVec,
    pub next_point: DVec,
    pub objF_next_point: f64,
    pub norm_gradient: f64,
}


impl fmt::Display for NewtonStep {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(format!(
            "moving to: {}\nold_trust_radius: {1:.5}, new_trust_radius: {2:.5},\n\
            r_ls: {3:.4}, r_cp: {4:.4}, r_dlp: {5:.4}, r_glm: {6:.4},\n\
            f(x_next): {7:.6},\n\
            ||gradient(f,x_next)||: {8:.4}\n\
            function value decrease (% current iterate):\
            ls: {9:.6}, cp: {10:.6}, dlp: {11:.6}, glm: {12:.6}\n\
            next point: {13:.4}",
            self.next_point_ID, self.old_trust_radius, self.new_trust_radius,
            self.r_ls, self.r_cp, self.r_dlp,self.r_glm,
            self.objF_next_point, self.norm_gradient,
            self.ls_decrease, self.cp_decrease, self.dlp_decrease, self.glm_decrease,
            self.next_point
        ).as_str())
    }
}


/// Cauchy point computed from current iterate x, gradient g and Hessian H of objective
/// function f at x.
///
fn cauchy_point(x:&DVec, g: &DVec, H: &DMat, r: f64) -> DVec {

    let norm_g_squared = g.norm_squared();
    let norm_g = norm_g_squared.sqrt();
    let Hg: DVec = H * g;
    let q = g.dot(&Hg);
    let t = (r / norm_g).min(norm_g_squared / q);

    x - t * g
}


/// Dog leg point computed from current iterate x, Cauchy point cp and directional
/// vector d in direction of the global quadratic minimizer b (solution of $(H+\lambda I)x = -g$.
///
fn dog_leg_point(x:&DVec,cp: &DVec, d: &DVec, r: f64) -> DVec {

    let u =(1f64/d.norm())*d;
    // solve ||x-(c+tu)||²=r², c=cauchy_point
    let c2 = (x-cp).norm_squared();  // ||x-c||²
    if c2>=r*r { return cp.clone(); }

    let cd = (x-cp).dot(&u);         // (x-c).u
    let sign = if cd > 0f64 { 1f64 } else { -1f64 };
    let t = cd - sign*(r*r - c2 + cd*cd).sqrt();
    (cp + t * u).clone()
}

/// Regularized global minimizer x+newton_step of quadratic approximation of f centered at x.
/// Newton_step: solution of (H+lambda*I) = -g
///
/// #Arguments:
///
/// 'g': gradient of f at x
/// 'H': Hessian of f at x
/// 'lambda': regularization parameter
///
pub fn global_quadratic_minimizer(x: &DVec, g: &DVec, H: &DMat, lambda:f64) -> Result<DVec> {

    let newton_step: DVec = solve_newton_equation(&g, &H,lambda)?;
    Ok(x+newton_step)
}

/// New trust radius for the next Newton step computed from the behaviour of f
/// versus the quadratic approximation of f in the current step.
///
/// #Arguments:
///
/// 'r': the old trust radius
/// 'fx': function value at current iterate
/// 'tp2': value of quadratic approximation of f at minimizer of f
/// 'points': list of points (r_u,f_u), u = one of the target points to move to
///           in the Newton step, r_u the distance of u from the current iterate
///           and f_u = f(u), the value of the objective function f at u.
///
type Point = (f64,f64);
use std::f64;
fn next_trust_radius(r:f64, fx:f64, tp2:f64, points: &Vec<Point>) -> f64 {

    // minimal value of f at the given points
    let f_min = points.iter().
        map(|(r_u,f_u):&(f64,f64)| -> f64 { *f_u }).
        fold(1e10,|a:f64, b:f64| a.min(b));

    // we go out to the farthest point achieving at least 5% of optimal decrease in f
    let r_new = points.iter().map(
        |(r_u,f_u):&(f64,f64)| -> f64 {
            if (fx-*f_u) >= 0.05*(fx-f_min) { *r_u } else { 0f64 }
        }).fold(0f64,|a, b| a.max(b));

    // modify the old radius based on ratio actual_decrease/quadratic_approx_decrease
    let q = (fx-f_min)/(fx-tp2);
    let r1 = if q > 0.6 { 1.5*r } else if q > 0.05 { r } else { r/1.5f64 };
    r1.max(r_new)
}



/// Returns Newton step to Dog-Leg point as Result<NewtonStep>
///
/// # Arguments
///
/// * `x`: current iterate
/// * `f`: objective function
/// * `r`: trust radius
/// * `g`: gradient at current iterate
/// * `H`: Hessian at current iterate
/// * `G`: region to which all points are confined
///       (e.g.: domain of definition of objective function or feasible set).
/// * `lambda`: regularization parameter in Newton equation.
///
pub fn newton_step(
    x: &DVec, min_prob: &dyn MinProblem, r:f64, lambda:f64
) -> Result<NewtonStep> {

    let G = min_prob.domain();
    assert!(G.contains(x),"iterate x not in region G = {}",G.id());

    let g = min_prob.gradient(&x);
    let H = min_prob.hessian(&x);

    let mut next_point_id: &str;
    let cp= cauchy_point(x,&g,&H,r);
    let cp_G: DVec = G.retract(x,&cp);
    let r_cp = (x-&cp_G).norm();

    // global minimizer glm of quadratic approximation:
    let glm = global_quadratic_minimizer(x,&g,&H,lambda)?;
    let mut glm_G: DVec = G.retract(x,&glm);
    let mut r_glm = (x-&glm_G).norm();

    // line search in direction of newton step
    let p = &glm_G-x;   // note: shorter than newton step because of retraction
    let f = |z: &DVec| min_prob.objective_fn(z);
    let phi = |t:f64| f(&(x+t*&p));
    let ls_result = golden_search(&phi,0f64,1.5f64,0.1f64);
    let t_ls = ls_result.0;
    let ls: DVec = x+t_ls*&p;            // minimizer of f in direction of glm
    let r_ls: f64 = t_ls*&p.norm();           // ||ls-x||
    let f_ls = ls_result.1;

    // dog-leg point, useful ony if b is outside the trust radius
    let d = &glm-&cp;    // direction from cp to glm
    let dlp: DVec = dog_leg_point(x,&cp, &d, r);
    let dlp_G: DVec = G.retract(x,&dlp);
    let r_dlp = (x-&dlp_G).norm();

    let fx = f(&x);
    // determine the point z=ls,glm,cp,dlp with minimal value f(z)
    let f_glm = f(&glm_G);
    let f_cp = f(&cp_G);
    let f_dlp = f(&dlp_G);
    let next_point: DVec =
        if f_ls.min(f_glm).min(f_cp).min(f_dlp) >= fx { // no decrease

            next_point_id = "no move";
            x.clone()

        } else if f_ls <= f_glm.min(f_cp).min(f_dlp) {

            next_point_id = "line search point";
            ls.clone()

        } else if f_glm <= f_ls.min(f_cp).min(f_dlp) {

            next_point_id = "global minimizer";
            glm_G.clone()

        } else if f_cp <= f_ls.min(f_glm).min(f_dlp) {

            next_point_id = "Cauchy point";
            cp_G

        } else {  // f_dlp is the smallest

            next_point_id = "dog leg point";
            dlp_G
        };

    let rho = 1e-10+fx.abs();
    let ls_decrease= 100f64*(fx-f_ls)/rho;
    let cp_decrease= 100f64*(fx-f_cp)/rho;
    let dlp_decrease= 100f64*(fx-f_dlp)/rho;
    let glm_decrease= 100f64*(fx-f_glm)/rho;

    // new trust radius
    // tp2: quadratic approximation of f centered at next point
    let h = &next_point;
    let tp2 = fx + (&g.dot(h) + 0.5f64*(&H * h).dot(h));

    let points = vec![(r_ls,f_ls),(r_glm,f_glm),(r_cp,f_cp),(r_dlp,f_dlp)];
    let new_trust_radius = next_trust_radius(r,fx,tp2,&points);


    let norm_grad = min_prob.gradient(&next_point).norm();
    let f_next_point = min_prob.objective_fn(&next_point);

    Ok( NewtonStep {
        next_point_ID: next_point_id,
        old_trust_radius: r,
        new_trust_radius,
        r_ls,
        r_cp,
        r_dlp,
        r_glm,
        ls_decrease,
        cp_decrease,
        dlp_decrease,
        glm_decrease,
        current_point: x.clone(),
        next_point,
        objF_next_point: f_next_point,
        norm_gradient: norm_grad,
    })
}

