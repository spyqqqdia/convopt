use convopt::{
    DVec, DMat,
    matrix_utils::*,
    equation::*
};
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::SeedableRng};

fn main() {

    let mut rng:Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(37);
    for k in 1..10 {

        let l_max = (k as f64)*100f64;     // largest eigenvalue
        let l_min = 0.001f64;              // smallest eigenvalue
        let lambda = 0.0000001f64;         // regularization parameter
        let n = k*100usize;

        let A = random_psd_matrix(n,l_min,l_max,&mut rng);
        // first equilibrate, then apply regularization!
        let B = ruiz_equilibration(&A,5,5).1;  // as used in cholesky-solve
        let b = random_vector(n,0f64,1f64,&mut rng);

        println!("\n\nSolving Ax=b with random {}x{} psd matrix A, vector b, lambda = {}:",
                 n,n,lambda);
        // 0:.1 first argument with precision=1
        println!("||b||_2 = {0:.1}, ||A||_2 = {1:.1}, cond(A) = {2:.0}",
                 &b.norm(),&A.norm(),l_max/l_min);
        println!("\nSolution via regularization (B+lambda*I)=b, B=ruiz_equilibrated(A)\
                 using cholesky_solve:\n\
                 ||B||_2 = {0:.1}, cond(B+lambda*I) <= ||B||_2/lambda={1:.0}",
                 &B.norm(),&B.norm()/lambda
        );
        if let Ok(x) = cholesky_solve_regularized(&A, &b, lambda){
            let residual = &b - &A*x;
            println!("||b-Ax||_2 as percent of ||A||_2: {0:.1}", 100f64*&residual.norm()/&A.norm());
            println!("||b-Ax||_2 as percent of ||b||_2: {0:.1}", 100f64*&residual.norm()/&b.norm());
        } else {
            println!("Cholesky factorization failed.")
        }
        println!("\nSolution with no regularization or Ruiz equilibration\n via Cholesky \
                 factorization, forward_solve and back_solve:");
        let chol_A = A.clone().cholesky().unwrap();
        let L: DMat = chol_A.l();
        let U: DMat = L.transpose();
        let w= forward_solve(&L,&b,0f64).unwrap();
        let x = back_solve(&U,&w,0f64).unwrap();
        let residual = &b - &A*x;
        println!("||b-Ax||_2 as percent of ||A||_2: {0:.1}", 100f64*&residual.norm()/&A.norm());
        println!("||b-Ax||_2 as percent of ||b||_2: {0:.1}", 100f64*&residual.norm()/&b.norm());
    }
}

