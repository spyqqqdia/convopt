use convopt::{
    DVec,
    optimization::{solve_min_problem},
    test_problems::Rosenbrook
};

fn main() {

    let min_prob = Rosenbrook::new(1f64,10f64);
    let eps = 1e-4;
    let max_iter = 100usize;
    println!("\nSolving problem Rosenbrook, minimum at (x,y)=(1,-1):");
    match solve_min_problem(&min_prob, eps, max_iter) {
        Ok(x) => println!("Solution:\n{}", x),
        Err(e) => println!("Error: {}", e)
    }
}