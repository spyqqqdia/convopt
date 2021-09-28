
use std::{error::Error, fmt};
use std::fmt::{Debug, Formatter, Display};


/// Possible errors occurring in this library.
#[derive(Debug, Clone)]
pub struct ConvOptError {
    pub kind: ErrKind,
}

#[derive(Debug, Clone)]
pub enum ErrKind {

    BackSolveFailure(&'static str),
    ForwardSolveFailure(&'static str),
    CholeskyFailure(&'static str),
    QRSolveFailure(&'static str),
    ConvergenceFailure(&'static str),
}





impl ConvOptError {

    pub(crate) fn new(kind: ErrKind) -> ConvOptError {

        ConvOptError{ kind,}
    }
}

impl fmt::Display for ConvOptError {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        match self.kind {

            ErrKind::BackSolveFailure(msg) => {
                let s = "Back solve failed: ".to_owned() + msg;
                f.write_str(s.as_str())
            },
            ErrKind::ForwardSolveFailure(msg) => {
                let s = "Forward solve failed: ".to_owned() + msg;
                f.write_str(s.as_str())
            },
            ErrKind::CholeskyFailure(msg) => {
                let s = "Cholesky factorization failed: ".to_owned() + msg;
                f.write_str(s.as_str())
            },
            ErrKind::ConvergenceFailure(msg) => {
                let s = "Convergence failed: ".to_owned() + msg;
                f.write_str(s.as_str())
            }
            ErrKind::QRSolveFailure(msg) => {
                let s = "qr_solve failed: ".to_owned() + msg;
                f.write_str(s.as_str())
            }
        }
    }
}

impl Error for ConvOptError {}
