use convopt::{
    DVec,
    optimization::{solve_min_problem},
    test_problems::Maxent
};

fn main() {
    let dim = 5usize;
    let min_prob = Maxent::new(dim);
    let eps = 1e-4;
    let max_iter = 100usize;
    println!("\nSolving problem Maxent in dimension {}", dim);
    match solve_min_problem(&min_prob, eps, max_iter) {
        Ok(x) => println!("Solution:\n{}", x),
        Err(e) => println!("Error: {}", e)
    }
}