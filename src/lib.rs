extern crate nnv_rs;
extern crate numpy;
extern crate pyo3;
extern crate rand;

use log::info;
use numpy::ndarray::Array1;
use numpy::Ix2;
use numpy::PyArray1;
use numpy::PyArray2;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::wrap_pyfunction;
use pyo3::PyObjectProtocol;
use serde_json;

use crate::nnv_rs::starsets::AdversarialStarSet2;
use crate::nnv_rs::starsets::Asterism;
use crate::nnv_rs::starsets::CensoredProbStarSet;
use crate::nnv_rs::starsets::CensoredProbStarSet2;
use crate::nnv_rs::starsets::ProbStarSet2;
use crate::nnv_rs::starsets::StarSet;
use crate::nnv_rs::starsets::VecStarSet;
use itertools::izip;
use log4rs::{
    append::file::FileAppender,
    config::{Appender, Config, Root},
    encode::pattern::PatternEncoder,
};
use nnv_rs::bounds::Bounds1;
use nnv_rs::deeppoly::deep_poly;
use nnv_rs::dnn::{DNNIndex, DNNIterator, Dense, ReLU, DNN};
use nnv_rs::star::Star2;
use rand::thread_rng;
use statrs::distribution::{ContinuousCDF, Normal};
use std::time::{Duration, Instant};

#[pyclass]
#[derive(Clone, Debug)]
struct PyBounds1 {
    bounds: Bounds1,
}

#[pymethods]
impl PyBounds1 {
    fn diag_gaussian_cdf(&self, mean: PyReadonlyArray1<f64>, scale: PyReadonlyArray1<f64>) -> f64 {
        let mut product_cdf = 1.;
        for (l, u, m, s) in izip!(
            self.bounds.lower(),
            self.bounds.upper(),
            mean.as_array(),
            scale.as_array()
        ) {
            let dim_norm = Normal::new(*m, *s).unwrap();
            let val = dim_norm.cdf(*u) - dim_norm.cdf(*l);
            product_cdf *= val;
        }
        product_cdf
    }
}

#[pyproto]
impl PyObjectProtocol for PyBounds1 {
    fn __str__(&self) -> String {
        format!("Bounds: {}", self.bounds)
    }
}

#[pyclass]
#[derive(Clone, Debug)]
struct PyDNN {
    dnn: DNN,
}

#[pymethods]
impl PyDNN {
    #[new]
    fn new() -> Self {
        Self {
            dnn: DNN::default(),
        }
    }

    pub fn input_shape(&self) -> Vec<Option<usize>> {
        self.dnn.input_shape().into()
    }

    fn add_dense(&mut self, filters: PyReadonlyArray2<f32>, bias: PyReadonlyArray1<f32>) {
        self.dnn.add_layer(Box::new(Dense::from_parts(
            filters
                .as_array()
                .to_owned()
                .mapv(|x| <f64 as num::NumCast>::from(x).unwrap()),
            bias.as_array()
                .to_owned()
                .mapv(|x| <f64 as num::NumCast>::from(x).unwrap()),
        )));
    }

    /*
    fn add_conv(&mut self, filters: PyReadonlyArray4<f32>, bias: PyReadonlyArray1<f32>) {
        self.dnn.add_layer(Layer::new_conv(Affine4::new(
            filters
                .as_array()
                .to_owned()
                .mapv(|x| <f64 as num::NumCast>::from(x).unwrap()),
            bias.as_array()
                .to_owned()
                .mapv(|x| <f64 as num::NumCast>::from(x).unwrap()),
        )))
    }
    */

    //fn add_maxpool(&mut self, pool_size: usize) {
    //    self.dnn.add_layer(Layer::new_maxpool(pool_size))
    //}

    //fn add_flatten(&mut self) {
    //    self.dnn.add_layer(Layer::Flatten)
    //}

    fn add_relu(&mut self, ndim: usize) {
        self.dnn.add_layer(Box::new(ReLU::new(ndim)))
    }

    fn deeppoly_output_bounds(
        &self,
        lower_input_bounds: PyReadonlyArray1<f64>,
        upper_input_bounds: PyReadonlyArray1<f64>,
    ) -> Py<PyTuple> {
        let input_bounds = Bounds1::new(
            lower_input_bounds.as_array().view(),
            upper_input_bounds.as_array().view(),
        );
        let output_bounds = deep_poly(
            &input_bounds,
            &self.dnn,
            DNNIterator::new(&self.dnn, DNNIndex::default()),
        );
        let gil = Python::acquire_gil();
        let py = gil.python();
        let out_lbs = PyArray1::from_array(py, &output_bounds.lower());
        let out_ubs = PyArray1::from_array(py, &output_bounds.upper());
        PyTuple::new(py, &[out_lbs, out_ubs]).into()
    }
}

#[pyproto]
impl PyObjectProtocol for PyDNN {
    fn __str__(&self) -> String {
        format!("DNN: {}", self.dnn)
    }
}

#[pyclass]
struct PyStarSet {
    starset: VecStarSet<Ix2>,
}

#[pymethods]
impl PyStarSet {
    #[new]
    pub fn py_new(py_dnn: PyDNN) -> Self {
        let dnn = py_dnn.dnn;
        let input_shape = dnn.input_shape();
        let star = Star2::default(&input_shape);
        let starset = VecStarSet::new(dnn, star);
        PyStarSet { starset }
    }

    pub fn minimal_norm_targeted_attack_delta(
        &mut self,
        x: PyReadonlyArray1<f64>,
        target_y: usize,
    ) -> Py<PyArray1<f64>> {
        let delta = self
            .starset
            .minimal_norm_targeted_attack_delta(&x.as_array().to_owned(), target_y);
        let gil = Python::acquire_gil();
        let py = gil.python();
        PyArray1::from_array(py, &delta).to_owned()
    }
}

#[pyclass]
struct PyAsterism {
    asterism: Asterism<Ix2>,
}

#[pymethods]
impl PyAsterism {
    #[new]
    pub fn py_new(
        py_dnn: PyDNN,
        input_bounds: Option<(PyReadonlyArray1<f64>, PyReadonlyArray1<f64>)>,
        loc: PyReadonlyArray1<f64>,
        scale: PyReadonlyArray2<f64>,
        safe_value: f64,
        cdf_samples: usize,
        max_iters: usize,
        stability_eps: f64,
    ) -> Self {
        let dnn = py_dnn.dnn;
        let input_shape = dnn.input_shape();
        let bounds = input_bounds.map(|(lbs, ubs)| Bounds1::new(lbs.as_array(), ubs.as_array()));

        let star = match input_shape.rank() {
            1 => Star2::default(&input_shape),
            _ => {
                panic!()
            }
        };
        Self {
            asterism: Asterism::new(
                dnn,
                star,
                loc.as_array().to_owned(),
                scale.as_array().to_owned(),
                safe_value,
                bounds,
                cdf_samples,
                max_iters,
                stability_eps,
            ),
        }
    }

    pub fn serialize(&self) -> String {
        let serialized = serde_json::to_string(&self.asterism).unwrap();
        serialized
    }

    pub fn build_tree(&mut self, num_samples: usize) {
        let mut rng = thread_rng();
        self.asterism.dfs_samples(num_samples, &mut rng, None);
    }

    pub fn get_mean(&self) -> Py<PyArray1<f64>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        PyArray1::from_array(py, &self.asterism.get_loc()).to_owned()
    }

    pub fn set_mean(&mut self, val: PyReadonlyArray1<f64>) {
        self.asterism.set_loc(val.as_array().to_owned())
    }

    pub fn get_scale(&self) -> Py<PyArray2<f64>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        PyArray2::from_array(py, &self.asterism.get_scale()).to_owned()
    }

    pub fn set_scale(&mut self, val: PyReadonlyArray2<f64>) {
        self.asterism.set_scale(val.as_array().to_owned())
    }

    pub fn get_input_bounds(&self) -> Option<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let input_bounds = self
            .asterism
            .get_input_bounds()
            .as_ref()
            .map(Bounds1::as_tuple);
        let gil = Python::acquire_gil();
        let py = gil.python();
        input_bounds.map(|(l, u)| {
            (
                PyArray1::from_array(py, &l).to_owned(),
                PyArray1::from_array(py, &u).to_owned(),
            )
        })
    }

    pub fn set_input_bounds(
        &mut self,
        fixed_part: Option<PyReadonlyArray1<f64>>,
        unfixed_part: Option<(PyReadonlyArray1<f64>, PyReadonlyArray1<f64>)>,
    ) {
        let fixed_bounds =
            fixed_part.map(|x| Bounds1::new(x.as_array().view(), x.as_array().view()));
        let unfixed_bounds =
            unfixed_part.map(|(l, u)| Bounds1::new(l.as_array().view(), u.as_array().view()));
        let bounds = match (fixed_bounds, unfixed_bounds) {
            (Some(f), Some(u)) => Some(f.append(&u)),
            (Some(f), None) => Some(f),
            (None, Some(u)) => Some(u),
            (None, None) => None,
        };
        let star = Star2::default(&self.asterism.get_dnn().input_shape());
        self.asterism.reset_with_star(star, bounds);
    }

    pub fn get_safe_value(&self) -> f64 {
        self.asterism.get_safe_value()
    }

    pub fn set_safe_value(&mut self, val: f64) {
        self.asterism.set_safe_value(val);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bounded_sample_input_multivariate_gaussian(
        &mut self,
        num_samples: usize,
        time_limit: Option<u64>,
    ) -> Option<(Py<PyArray1<f64>>, f64, f64)> {
        let mut rng = thread_rng();
        let output = self.asterism.sample_safe_star(
            num_samples,
            &mut rng,
            time_limit.map(Duration::from_millis),
        );
        let (samples, path_logp, invalid_cdf_proportion) = output.as_ref()?;
        let gil = Python::acquire_gil();
        let py = gil.python();
        Some((
            PyArray1::from_array(py, &samples[0]).to_owned(),
            *path_logp,
            *invalid_cdf_proportion,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_samples_and_overapproximated_infeasible_input_regions(
        &mut self,
        total_samples: usize,
        num_intermediate_samples: usize,
        time_limit_opt: Option<u64>,
    ) -> Option<(Vec<Vec<Py<PyArray1<f64>>>>, Vec<PyBounds1>)> {
        let start_time = Instant::now();
        info!("get_samples_and_overapproximated_infeasible_input_regions START");
        let mut rng = thread_rng();
        let mut sum_samples = 0;
        // Sample
        let sample_chunks: Vec<Vec<Array1<f64>>> = {
            let mut chunks = vec![];
            while sum_samples < total_samples {
                if let Some(chunk) = self.asterism.sample_safe_star(
                    num_intermediate_samples,
                    &mut rng,
                    time_limit_opt.map(Duration::from_millis),
                ) {
                    sum_samples += chunk.0.len();
                    chunks.push(chunk.0);
                } else {
                    info!("get_samples_and_overapproximated_infeasible_input_regions STOP: no feasible actions");
                    return None;
                }
            }
            chunks
        };
        // Invalidate
        self.asterism.dfs_samples(
            num_intermediate_samples,
            &mut rng,
            time_limit_opt
                .map(|x| Duration::saturating_sub(Duration::from_millis(x), start_time.elapsed())),
        );

        // return invalid regions
        let regions: Vec<PyBounds1> = self
            .asterism
            .get_overapproximated_infeasible_input_regions()
            .into_iter()
            .map(|bounds| PyBounds1 {
                bounds: bounds.unfixed_dims(),
            })
            .collect();
        let gil = Python::acquire_gil();
        let py = gil.python();
        let py_sample_chunks: Vec<Vec<Py<PyArray1<f64>>>> = sample_chunks
            .into_iter()
            .map(|chunk| {
                chunk
                    .into_iter()
                    .map(|x| PyArray1::from_array(py, &x).to_owned())
                    .collect()
            })
            .collect();
        info!("get_samples_and_overapproximated_infeasible_input_regions STOP: complete");
        Some((py_sample_chunks, regions))
    }
}

#[pyfunction]
fn start_logging(file_path: &str) {
    let level = log::LevelFilter::Info;
    let logfile = FileAppender::builder()
        // Pattern: https://docs.rs/log4rs/*/log4rs/encode/pattern/index.html
        .encoder(Box::new(PatternEncoder::new("{d} - {m}\n")))
        .build(file_path)
        .unwrap();
    let config = Config::builder()
        .appender(Appender::builder().build("logfile", Box::new(logfile)))
        .build(Root::builder().appender("logfile").build(level))
        .unwrap();
    let _handle = log4rs::init_config(config);
}

#[pyfunction]
fn halfspace_gaussian_cdf(
    coeffs: PyReadonlyArray1<f64>,
    rhs: f64,
    mu: PyReadonlyArray1<f64>,
    sigma: PyReadonlyArray1<f64>,
) -> f64 {
    let mu = mu.as_array().to_owned();
    let sigma = sigma.as_array().to_owned();
    let coeffs = coeffs.as_array().to_owned();
    nnv_rs::trunks::halfspace_gaussian_cdf(coeffs, rhs, &mu, &sigma)
}

/// # Errors
#[pymodule]
pub fn nnv_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyAsterism>()?;
    m.add_class::<PyDNN>()?;
    m.add_class::<PyStarSet>()?;
    m.add_function(wrap_pyfunction!(halfspace_gaussian_cdf, m)?)?;
    m.add_function(wrap_pyfunction!(start_logging, m)?)?;
    Ok(())
}
