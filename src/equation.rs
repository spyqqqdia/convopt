use crate::{
    Result, DVec, DMat,
    error::*,
    matrix_utils::*
};




/// Solve the regularized equation $(H+l*I)x=b$ by Cholesky factorization of
/// $H+l*I$. The matrix $H+l*I$ needs to be positive definite, i.e. if $H$ is singular
/// we must have $l>0$. Uses Ruiz preconditioning of $H+l*I$.
///
/// # Arguments
///
/// * `H`: positive semidefinite symmetric square matrix
/// * `l`: nonnegative scalar (regularization parameter)
/// * `b`: vector of same dimension as H
///
pub fn cholesky_solve_regularized(H: &DMat, b: &DVec, l: f64) -> Result<DVec> {

    let n = H.shape().0;  // number of rows
    assert!(n==H.shape().1 && n==b.len() && l>= 0f64);

    // ruiz equilibration to improve the condition number
    let n_e: usize = 5;   // rounds of ||.||_oo and ||.||_2 equilibration
    let (d,B) = ruiz_equilibration(H,n_e,n_e);
    // regularization
    let G: DMat = if l <= 0f64 { B } else { B + l*DMat::identity(n,n) };

    if let Some(ch) = G.cholesky() {

        let c = DVec::from_fn(n,|i,_| b[i]*d[i]);
        let u = ch.solve(&c);
        Ok(DVec::from_fn(n,|i,_| u[i]*d[i]))
    } else {
        Err(ConvOptError::new(ErrKind::CholeskyFailure("in cholesky_solve")))
    }
}



/// Solve the regularized equation $(H+l*I)x=b$ by QR factorization of
/// $H+l*I$. Uses Ruiz preconditioning of $H+l*I$.
///
/// # Arguments
///
/// * `H`: square matrix
/// * `l`: nonnegative scalar (regularization parameter)
/// * `b`: vector of same dimension as H
///
pub fn qr_solve(H: &DMat, b: &DVec, l: f64) -> Result<DVec> {

    let n = H.shape().0;  // number of rows
    assert!(n==H.shape().1 && n==b.len() && l>= 0f64);

    // first equilibrate, then apply regularization!
    let n_e: usize = 5;   // rounds of ||.||_oo and ||.||_2 equilibration
    let (d,B) = ruiz_equilibration(H,n_e,n_e);
    // regularization
    let G: DMat = if l <= 0f64 { B } else { B + l*DMat::identity(n,n) };

    let qr = G.qr();
    let c = DVec::from_fn(n,|i,_| b[i]*d[i]);
    if let Some(u) = qr.solve(&c) {
        Ok(DVec::from_fn(n,|i,_| u[i]*d[i]))
    } else {
        Err(ConvOptError::new(ErrKind::QRSolveFailure("in qr_solve")))
    }
}

/// Assuming that L is a lower triangular matrix, solves (L+lI)x=y, where I s the identity matrix.
/// Only the lower triangular part of L is used, lower triangularity is not checked.
///
///
pub fn forward_solve(L: &DMat, y: &DVec, l:f64) -> Result<DVec> {

    let n=L.shape().0;
    assert!(L.shape().1==n && y.len()==n);
    let normL = L.norm();
    let mut x = DVec::repeat(n,0f64);
    let mut i=0;
    while i<n {

        if (l+L[(i,i)]).abs()<1e-20*normL {
            return Err(ConvOptError::new(ErrKind::ForwardSolveFailure("Zero diagonal element")));
        }
        let mut s = 0f64;  // \sum_{j<i}L_ijx_j
        let mut j=0;
        while j< i { s+=L[(i,j)]*x[j]; j+=1; }

        x[i] = (y[i]-s)/(l+L[(i,i)]);
        i+=1;
    }
    Ok(x)
}


/// Assuming that U is an upper triangular matrix, solves (U+lI)x=y, where I s the identity matrix.
/// Only the upper triangular part of U is used, upper triangularity is not checked.
///
pub fn back_solve(U: &DMat, y: &DVec, l:f64) -> Result<DVec> {

    let n = U.shape().0;
    assert!(U.shape().1==n && y.len()==n);
    let normL = U.norm();
    let mut x = DVec::repeat(n,0f64);
    let mut i= n-1;
    let mut go_on = true;    // checks i>=0
    while go_on {

        if (l+U[(i,i)]).abs()<1e-20*normL {
            return Err(ConvOptError::new(ErrKind::BackSolveFailure("Zero diagonal element")));
        }
        let mut s = 0f64;  // \sum_{j>}U_ijx_j
        let mut j = n-1;
        while j>i { s+=U[(i,j)]*x[j]; j-=1; }

        x[i] = (y[i]-s)/(l+U[(i,i)]);
        // note i-=1 will fail for i==0 because of i:usize
        if i==0 { go_on=false; } else { i-=1} ;
    }
    Ok(x)
}


