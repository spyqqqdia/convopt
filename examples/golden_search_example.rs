use convopt::optimization::golden_search;

fn main() {

    // P(t)=2(t-1)+3(t-1)² has its minimum  at t=2/3
    let mut f = |t:f64| 2f64*(t-1f64) + 3f64*(t-1f64)*(t-1f64);
    let mut a=-1f64;
    let mut b = 2f64;
    let eps = 0.1;
    let res = golden_search(&f,a,b,eps);
    println!("\nGolden search result for P(t)=2(t-1)+3(t-1)² on [-1,2] with eps=0.1\n\
             minimizer is t=2/3:\n\
             a = {0:.4}, f(a) = {1:.4}\n",res.0,res.1
    );

    let f = |t:f64| t*t.ln()+(1f64-t)*(1f64-t).ln();
    let a = 0.0001f64;
    let b = 0.9999f64;
    let eps = 0.1;
    let res = golden_search(&f,a,b,eps);
    println!("\nGolden search result for f(t)= t*ln(t)+(1-t)*ln(1-t) on (0,1) with eps=0.1\n\
             minimizer is t=1/2:\n\
             a = {0:.4}, f(a) = {1:.4}\n",res.0,res.1
    );

}