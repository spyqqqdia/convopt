use convopt::{
    DVec,
    optimization::{solve_min_problem},
    test_problems::Rosenbrook
};

fn main() {
    let dim = 5usize;
    let min_prob = Rosenbrook::new(1f64,100f64);
    let eps = 1e-4;
    let max_iter = 100usize;
    println!("\nSolving problem Rosenbrook:");
    match solve_min_problem(&min_prob, eps, max_iter) {
        Ok(x) => println!("Solution:\n{}", x),
        Err(e) => println!("Error: {}", e)
    }
}