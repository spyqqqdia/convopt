use convopt::{
    matrix_utils::*,
    equation::*
};
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::SeedableRng};

fn main() {

    let mut rng:Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(37);

    for k in 1..10 {

        let l_max = (k as f64)*100f64;  // largest eigenvalue
        let l_min = 0.001f64;           // smallest eigenvalue
        let lambda = 0.00001f64;         // regularization parameter
        let n = k*100usize;

        let A = random_psd_matrix(n,l_min,l_max,&mut rng);
        let B = ruiz_equilibration(&A,5,5).1;  // as used in qr-solve
        let b = random_vector(n,0f64,1f64,&mut rng);

        println!("\nSolving Ax=b with random {}x{} psd matrix A, vector b, lambda = {}:",
                 n,n,lambda
        );
        // 0:.1 first argument with precision=1
        println!("||b||_2 = {0:.1}, ||A||_2 = {1:.1}, cond(A) = {2:.0}, ||B||_2 = {3:.1}",
            &b.norm(),&A.norm(),l_max/l_min,&B.norm()
        );

        if let Ok(x) = qr_solve(&A,&b,lambda){
            let residual = &b - &A*x;
            println!("cond(B+lambda*I) <= ||B||_2/lambda = {0:.0}",&B.norm()/lambda);
            println!("||b-Ax||_2 as percent of ||A||_2: {0:.1}", 100f64*&residual.norm()/&A.norm());
            println!("||b-Ax||_2 as percent of ||b||_2: {0:.1}", 100f64*&residual.norm()/&b.norm());
        } else {
            println!("qr_solve failed (matrix singular?).")
        }
    }
}