use std::fmt;
use crate::{
    error::ConvOptError, error::ErrKind,
    equation::cholesky_solve_regularized,
    Result, DVec, DMat, FUN_nD_TO_1D,
    logging::Logger,
    optimization::MinProblem
};

use super::Region;




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
    /// step size to cauchy point
    pub r_cp: f64,
    /// step size to dog leg point
    pub r_dlp: f64,
    /// step size to global quadratic optimizer
    pub r_glm: f64,
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
            "moving to: {}, old_trust_radius: {1:.5}, new_trust_radius: {2:.5},\n\
            r_cp: {3:.4}, r_dlp: {4:.4}, r_glm:{5:.4},\n\
            objF(next_point): {6:.6},\n\
            norm of gradient: {7:.4}\n\
            function value decrease (% current iterate): cp: {8:.6}, dlp: {9:.6}, glm: {10:.6}, \
            next point: {11:.4}",
            self.next_point_ID, self.old_trust_radius, self.new_trust_radius,
            self.r_cp, self.r_dlp,self.r_glm,
            self.objF_next_point, self.norm_gradient,
            self.cp_decrease, self.dlp_decrease, self.glm_decrease, &self.next_point
        ).as_str())
    }
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
    x: &DVec, min_prob: &MinProblem, r:f64, lambda:f64
) -> Result<NewtonStep> {

    let G = min_prob.domain();
    assert!(G.contains(x),"iterate x not in region G = {}",G.id());

    let g = min_prob.gradient(&x);
    let norm_g_squared = g.norm_squared();
    let norm_g = norm_g_squared.sqrt();
    let H = min_prob.hessian(&x);
    let Hg: DVec = &H * &g;
    let q = &g.dot(&Hg);
    let t = (r / norm_g).min(norm_g_squared / q);

    let next_point_id: &str;
    let new_trust_radius: f64;
    let cp: DVec = x - t * &g;  // Cauchy point
    let cp_G: DVec = G.retract(x,&cp);
    let r_cp = (x-&cp_G).norm();

    // global minimizer glm of quadratic approximation:
    let newton_step: DVec = solve_newton_equation(&g, &H,lambda)?;
    let glm = x+newton_step;
    let mut glm_G: DVec = G.retract(x,&glm);
    let mut r_glm = (x-&glm_G).norm();

    // dog-leg point, useful ony if b is outside the trust radius
    let d = &glm-&cp;    // direction from cp to glm
    let dlp: DVec = dog_leg_point(x,&cp, &d, r);
    let dlp_G: DVec = G.retract(x,&dlp);
    let r_dlp = (x-&dlp_G).norm();


    let fx = min_prob.objective_fn(&x);
    let f_cp= min_prob.objective_fn(&cp_G);
    let f_glm = min_prob.objective_fn(&glm_G);
    let f_dlp=  min_prob.objective_fn(&dlp_G);

    let rho = 1e-10+fx.abs();
    let cp_decrease = 100f64*(fx-f_cp) / rho;
    let dlp_decrease = 100f64*(fx-f_dlp) / rho;
    let glm_decrease = 100f64*(fx-f_glm) / rho;

    // if the glm solution is closer the the current iterate than the cp but worse,
    // maybe the regularization parameter lambda was too big, shrink it:
    let glm_step_too_small =
        (r_glm<=r_cp) && (glm_decrease < cp_decrease) ||
            (r_glm<=r_dlp) && (glm_decrease < dlp_decrease);
    if glm_step_too_small {

        let bigger_newton_step = solve_newton_equation(&g, &H,lambda/10f64)?;
        let glm_1 = x+bigger_newton_step;
        glm_G = G.retract(x,&glm_1); // the old glm is irrelevant
        r_glm = (x-&glm_G).norm();
    }

    let next_point: DVec =
        if f_cp.min(f_glm).min(f_dlp) >= fx { // no decrease

            new_trust_radius = r/2f64;
            next_point_id = "no move";
            x.clone()

        } else if f_cp <= f_glm.min(f_dlp) {

            next_point_id = "Cauchy point";
            new_trust_radius = (r_cp+r)/2f64;
            cp_G

        } else if f_glm <= f_cp.min(f_dlp) {

            new_trust_radius = 0.1f64*r+0.9f64*r_glm;
            next_point_id = "global minimizer";
            glm_G.clone()

        } else {  // f_dlp is the smallest

            new_trust_radius = 1.5f64*r;
            next_point_id = "dog leg point";
            dlp_G
        };

    let norm_grad = min_prob.gradient(&next_point).norm();
    let f_next_point = min_prob.objective_fn(&next_point);

    Ok( NewtonStep {
        next_point_ID: next_point_id,
        old_trust_radius: r,
        new_trust_radius,
        r_cp,
        r_dlp,
        r_glm,
        cp_decrease,
        dlp_decrease,
        glm_decrease,
        current_point: x.clone(),
        next_point,
        objF_next_point: f_next_point,
        norm_gradient: norm_grad,
    })
}

