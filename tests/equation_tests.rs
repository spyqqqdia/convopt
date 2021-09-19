use convopt::{
    DVec, DMat,
    matrix_utils::*,
    equation::*
};
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::SeedableRng};

#[test]
fn test_cholesky_solve() {

    println!("Starting cholesky solve test:");
    let mut rng:Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(37);
    let l_max = 1000f64;
    let l_min = 0.001f64;
    let n = 1000usize;
    let A = random_psd_matrix(n,l_min,l_max,&mut rng);
    let b = random_vector(n,0f64,1f64,&mut rng);

    println!("Solution via cholesky_solve:");
    if let Ok(x) = cholesky_solve_regularized(&A, &b, 0f64){
        let residual = &b - &A*x;
        assert!(residual.norm() < 1e-12*&A.norm());
    } else {
        assert!(false,"Cholesky decomposition failed");
    }
    println!("Solution via Cholesky factorization, forward_solve and back_solve:");
    let chol_A = A.cholesky().unwrap();
    let L: DMat = chol_A.l();
    let U: DMat = L.transpose();
    let w = forward_solve(&L,&b,0f64)?;
    let x = back_solve(&U,&w,0f64)?;
    let residual = &b - &A*x;
    assert!(residual.norm() < 1e-12*&A.norm());
}