pub mod constants;
pub mod structs;

pub use structs::{GLFunction, GLWitness};

pub fn check_gl_quad <T: GLFunction> (wit: GLWitness<T>) {
    wit.wit_struct.check_y_bound(wit.inp_struct);
    wit.wit_struct.check_ratio(wit.inp_struct);
    wit.wit_struct.check_mult(wit.inp_struct);
    wit.wit_struct.check_inverse(wit.inp_struct);
    wit.wit_struct.check_sum(wit.inp_struct);
}