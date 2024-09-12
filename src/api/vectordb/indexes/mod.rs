use actix_web::{web, Scope};
use controller::create_index;

mod controller;
mod service;
mod dtos;

pub(crate) fn indexes_module() -> Scope {
    let indexes_module = web::scope("/indexes").route("", web::post().to(create_index));

    indexes_module
}