use convopt::optimization::golden_search;

fn main() {

    // P(t)=2(t-1)+3(t-1)² has its minimum  at t=2/3
    let f = |t:f64| 2f64*(t-1f64) + 3f64*(t-1f64)*(t-1f64);
    let a=-1f64;
    let b = 2f64;
    let eps = 0.1;
    let res = golden_search(f,a,b,eps);
    println!("\nGolden search result for P(t)=2(t-1)+3(t-1)² on [-1,2] with eps=0.1\n\
             minimizer is t=2/3:\n\
                a = {0:.2}, P(a) = {1:.2},\n\
                b = {2:.2}, P(b) = {3:.2},\n\
                c = {4:.2}, P(c) = {5:.2}\n",
       res.0,res.1,res.2,res.3,res.4,res.5
    );

}