#![crate_name = "convopt"]

/// This crate provides the tools to perform convex optimization under constraints.
/// Both the objective function as well as the constraints are assumed to be convex,
/// twice continuously differentiable and the gradient and Hessian are assumed to be available.
///
///
///
///
///
///

#[macro_use]
extern crate rand_xoshiro;
extern crate rand;
extern crate approx; // For the macro relative_eq!
extern crate nalgebra;
extern crate plotlib;
extern crate lazysort;



pub(crate) use nalgebra::base::{
    DMatrix, DVector
};


/// A `Result` where the `Err` case is `crate::error::Error`.
pub type Result<T> = std::result::Result<T, self::error::ConvOptError>;
pub type DVec = DVector<f64>;
pub type DMat = DMatrix<f64>;
pub type FUN_nD_TO_1D = fn(&DVec) -> f64;


pub mod error;
pub mod logging;
pub mod matrix_utils;
pub mod equation;
pub mod optimization;
pub mod test_problems;










