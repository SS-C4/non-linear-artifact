pub mod constants;
pub mod structs;

pub use structs::{LookupFunction, LookupWitness};

pub fn check_lookup <T: LookupFunction> (wit: LookupWitness<T>) {
    wit.wit_struct.check_lookup(wit.inp_struct);
}