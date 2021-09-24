use crate::{
    error::ConvOptError, error::ErrKind,
    equation::cholesky_solve_regularized,
    Result, DVec, DMat, FUN_nD_TO_1D,
    logging::Logger,
    optimization::MinProblem
};


/// Auxilliary function attemtps to locate minimum of closure f on
/// the interval [a,b] b performing golden search until the
/// a minimizer is bracketed in an interval of length at most eps.
///
/// # Arguments
///
/// * `fa` f(a)
/// * `fb` f(b)
/// * `fc` f(c)
/// * eps termination criterion: b-a < eps
///
/// Returns: (a1,f(a1),b1,f(b1),c1,f(c1)) with b1-a1 <= eps, a1<c1<b1 and
/// [a1,b1] brackets a local or global minimizer of f on [a,b].
///
/// This function is used to run the recursion in the golden search below.
///
pub fn golden_search_rec<F>(
    f: F, a: f64, fa: f64, b: f64, fb: f64, c:f64, fc: f64, eps: f64) -> (f64,f64,f64,f64,f64,f64)
where F: Fn(f64) -> f64
{
    assert!(a<c && c<b);

    if b-a<=eps { (a,fa,b,fb,c,fc) } else {

        let rho = (5f64.sqrt()-1f64)/2f64;
        if b-c > c-a {  // split the interval [c,b]

            let d = b-rho*(b-c);
            let fd = f(d);
            if fa > fd.min(fc.min(fb)) {

                // minimum is in [c,b] with c<d<b
                golden_search_rec(f,c,fc,b,fb,d,fd,eps)
            } else {

                // minimum is in [a,d] with a<c<d
                golden_search_rec(f,a,fa,d,fd,c,fc,eps)
            }

        } else {  // c-a > b-c, split the interval [a,c]

            let d = c-rho*(c-a);
            let fd = f(d);
            if fa > fd.min(fc.min(fb)) {

                // minimum is in [d,b] with d<c<b
                golden_search_rec(f,d,fd,b,fb,c,fc,eps)
            } else {

                // minimum is in [a,c] with a<d<c
                golden_search_rec(f,a,fa,c,fc,d,fd,eps)
            }
        }
    }
}


/// Auxilliary function attemtps to locate minimum of closure f on
/// the interval [a,b] b performing golden search until the
/// a minimizer is bracketed in an interval of length at most eps.
///
/// # Arguments
///
/// * `fa` f(a)
/// * `fb` f(b)
/// * `fc` f(c)
/// * eps termination criterion: b-a < eps
///
/// Returns: (a1,f(a1),b1,f(b1),c1,f(c1)) with b1-a1 <= eps, a1<c1<b1 and
/// [a1,b1] brackets a local or global minimizer of f on [a,b].
///
/// This function is used to run the recursion in the golden search below.
///
pub fn golden_search<F>(
    f: F, a: f64, b: f64, eps: f64) -> (f64,f64,f64,f64,f64,f64)
    where F: Fn(f64) -> f64
{
    let rho = (5f64.sqrt()-1f64)/2f64;
    let fa = f(a);
    let fb = f(b);
    let c = b-rho*(b-a);
    let fc = f(c);
    golden_search_rec(f,a,fa,b,fb,c,fc,eps)
}
