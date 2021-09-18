
use std::io::{stdout, Write};
use convopt::{
    DVec,
    matrix_utils::*,
    logging::Logger
};
use nalgebra::SymmetricEigen;
use lazysort::Sorted;
use rand_xoshiro::{Xoshiro256PlusPlus, rand_core::SeedableRng};

/// We allocate 200 random symmetric positive definite 500 x 500 matrices A with
/// condition number = 1e6,
/// apply the Ruiz equilibration algorithm and compute the condition number of the equilibrated
/// matrix B. The ratios cond(B) / cond(A) (hopefully << 1) are kept and a histogram of same
/// is printed, as well as some statistics written to a text file.
///
fn main() {

    let mut rng:Xoshiro256PlusPlus = Xoshiro256PlusPlus::seed_from_u64(37);

    let l_max = 1000f64;    // largest eigenvalue
    let l_min = 0.001f64;   // smallest eigenvalue
    let cond_A = l_max/l_min;
    let n = 200;
    let dim = 200usize;
    let mut cond_quotient: Vec<f64> =  Vec::with_capacity(n);

    let mut logger = Logger::new("results/ruiz_conditioning.txt");
    logger.write(format!(
        "\nStatistics for cond(ruiz_equilibrated(A))/cond(A) for {} {}x{} matrices:\n\n",
        n,dim,dim
    ).as_str());


    println!("\nComputing condition numbers of {} {}x{} matrices",n,dim,dim);
    println!("this can take a while (20 *): ");

    let n_oo: usize = 5;   // number of rounds of ||.||_oo equilibration
    let n_2: usize = 5;    // number of rounds of ||.||_2 equilibration

    for k in 0..n {

        if k%(n/20)==0 { print!("*"); stdout().flush(); }
        let A = random_psd_matrix(dim,l_max,l_min,&mut rng);
        let B = ruiz_equilibration(&A,n_oo,n_2).1;

        let eig_B = SymmetricEigen::new(B);
        let eig_vals_B = eig_B.eigenvalues;
        let cond_B = eig_vals_B.amax()/eig_vals_B.amin();
        cond_quotient.push(cond_B/cond_A);
    }
    cond_quotient.sort_by(|a:&f64,b:&f64| a.partial_cmp(b).unwrap());
    logger.write("\nStatistics for cond(ruiz_equilibrated(A))/cond(A):\n");
    logger.write(format!("quantile_10%: {}\n",cond_quotient[n/10]).as_str());
    logger.write(format!("quantile_50%: {}\n",cond_quotient[n/2]).as_str());
    logger.write(format!("quantile_90%: {}\n",cond_quotient[9*n/10]).as_str());
    println!("\nFinished, results in results/ruiz_conditioning.txt")
}
