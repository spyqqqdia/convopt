use rand::{prelude::*,Rng};
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::SeedableRng};
use rand_distr::{StandardNormal};
use crate::{DVec, DMat};



/// A vector in $R^dim$ with uniformly random f64 components in [a,b).
pub fn random_vector(dim:usize,a:f64,b:f64, rng: &mut impl Rng) -> DVec {

    DVec::from_fn(dim, |_i,_| a+(b-a)*rng.gen::<f64>())
}

/// An mxn matrix with uniformly random f64 components in [a,b).
pub fn random_matrix(m:usize,n:usize,a:f64,b:f64, rng: &mut impl Rng) -> DMat {

    DMat::from_fn(m,n, |_r,_c| a+(b-a)*rng.gen::<f64>())
}

/// A random nxn orthogonal matrix. The matrix will be distributed uniformly
/// with respect to Haar measure on the orthogonal group O(n).
pub fn random_orthogonal_matrix(n:usize, rng: &mut impl Rng) -> DMat {

    let a = DMat::from_fn(n,n, |_r,_c| StandardNormal.sample(rng));
    let qr_a = a.qr();
    qr_a.q()
}

/// A random symmetric nxn matrix A with prescribed eigenvalues, i.e. A = UDU', where
/// U is a random orthogonal matrix and D the diagonal matrix with diag(D) = eigvals,
/// the vector of prescribed eigen values.
///
pub fn random_symmetric_matrix(eigvals:DVec, rng: &mut impl Rng) -> DMat {

    let n = eigvals.shape().1;
    let u = random_orthogonal_matrix(n,rng);
    &u*(DMat::from_diagonal(&eigvals)*&u.transpose())
}

/// A random symmetric positive definite nxn matrix A with eigenvalues decreasing exponentially
/// from l_max > l_min to l_min > 0.
/// Note: this matrix has condition number l_max/l_min. This function is intended to provide
/// examples of ill conditioned matrices so we can test the efficiency of preconditioners or
/// the accuracy of equation solvers.
///
pub fn random_psd_matrix(n:usize,l_min:f64,l_max:f64, rng: &mut impl Rng) -> DMat {

    assert!(n>1);
    let q = (l_max/l_min).ln() / ((n-1) as f64);
    let eigvals: DVec = DVec::from_fn(n,|i:usize,_| l_max*(-((i as f64)*q)).exp());
    let u = random_orthogonal_matrix(n,rng);
    &u*(DMat::from_diagonal(&eigvals)*&u.transpose())
}
/// Cross product of the vectors u,v, i.e the (rank one) matrix (u_i*v_j)
///
pub fn cross_product(u:&DVec,v:&DVec) -> DMat {

    let m = u.len();
    let n = v.len();
    DMat::from_fn(m,n,|r,c| u[r]*v[c])
}




/// Computes a matrix B = DAD, where D=diag(d) is a diagonal matrix with positive diagonal d
/// such that the entries of B are more uniform in size. It is hoped that B has a smaller
/// condition number than A.
///
/// This is used when solving equations: the equation Ax=b is solved as follows: first multiply
/// by D to obtain the equivalent equation DAx=Db. Then set x=Du to transform it to Bu=Db.
/// Solve that equation for u and obtain x as x=Du.
///
/// # Arguments
///
/// * `n_oo`: number of iterations of ||.||_oo equilibration
/// * `n_2`:  number of iterations of ||.||_2 equilibration
///
/// Returns tuple (d,B).
///
pub fn ruiz_equilibration(A: &DMat,n_oo:usize,n_2:usize) -> (DVec, DMat){

    let n = A.shape().0;
    assert!(A.shape().1==n,"matrix A not square, rows={}, cols={}",n,A.shape().1);

    let mut d: DVec = DVec::repeat(n,1f64);
    let mut B = A.clone();
    let mut i = 0;
    let mut k = 0;

    // 30 rounds of ||.||_oo equilibration
    while k < n_oo {
        while i<n {

            let mut f_i = B.row(i).amax().sqrt();
            if f_i > 0f64 { d[i] /= f_i; }
            i += 1;
        }
        B = DMat::from_fn(n,n,|r,c| d[r]*d[c]*A[(r,c)]);
        k += 1;
    }


    // 30 rounds of ||.||_2 equilibration
    k = 0;
    while k < n_2 {

        i = 0;
        while i<n {

            let mut f_i = B.row(i).norm().sqrt();
            if f_i > 0f64 { d[i] /= f_i; }
            i += 1;
        }
        B = DMat::from_fn(n,n,|r,c| d[r]*d[c]*A[(r,c)]);
        k += 1;
    }
    (d,B)
}