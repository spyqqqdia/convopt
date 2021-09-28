use crate::{
    error::ConvOptError, error::ErrKind,
    equation::cholesky_solve_regularized,
    Result, DVec, DMat, FUN_nD_TO_1D,
    logging::Logger,
    optimization::MinProblem
};

/// Minimizes the quadratic polynomial interpolating the points
/// (a,fa), (b,fb) and (c,fc), returns the point (u,f(u)) at which
/// the minimum of f occurs among the points examined.
///
pub fn poly2min<F>(f: &F, a:f64, fa:f64, b:f64, fb:f64, c:f64, fc:f64) -> (f64,f64)
where F: Fn(f64) -> f64
{
    // current minimum
    let (q,fq) =
        if fa<=fb.min(fc) { (a,fa) } else
        if fb<=fa.min(fc) { (b,fb) } else { (c,fc) };

    // docs/PD.pdf, p20, eq(1.37)
    let N: f64 = fa*(b*b-c*c) + fb*(c*c-a*a) + fc*(a*a-b*b);
    let D: f64 = fa*(b-c) + fb*(c-a) + fc*(a-b);
    let u: f64 = 0.5f64*(N/D);

    // D <= 0: quadratic polynomial has a max not a min
    if D<=0f64 || u<a || u> b { (q,fq) } else {

        let fu = f(u);
        if fu <= fq { (u,fu) } else { (q,fq) }
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
pub fn golden_search_rec<F>(
    f: &F, a: f64, fa: f64, b: f64, fb: f64, c:f64, fc: f64, eps: f64) -> (f64,f64,f64,f64,f64,f64)
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
/// Returns: (u,fu) where u is the local or global minimizer of f on [a,b]
/// computed.
///
pub fn golden_search<F>(
    f: &F, a: f64, b: f64, eps: f64) -> (f64,f64)
    where F: Fn(f64) -> f64
{
    let rho = (5f64.sqrt()-1f64)/2f64;
    let fa = f(a);
    let fb = f(b);
    let c = b-rho*(b-a);
    let fc = f(c);
    let gs = golden_search_rec(f,a,fa,b,fb,c,fc,eps);
    poly2min(f,gs.0,gs.1,gs.2,gs.3,gs.4,gs.5)
}
