extern crate env_logger;
extern crate nnv_rs;
extern crate numpy;
extern crate pyo3;
extern crate rand;

use numpy::PyArray1;
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::PyObjectProtocol;

use nnv_rs::affine::{Affine2, Affine4};
use nnv_rs::asterism::Asterism;
use nnv_rs::bounds::{Bounds, Bounds1};
use nnv_rs::constellation::Constellation;
use nnv_rs::deeppoly::deep_poly;
use nnv_rs::dnn::{DNNIndex, DNNIterator, Layer, DNN};
use nnv_rs::star::Star2;
use numpy::Ix2;
use rand::thread_rng;
use std::time::Duration;

#[pyclass]
#[derive(Clone, Debug)]
struct PyDNN {
    dnn: DNN<f64>,
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
        self.dnn.add_layer(Layer::new_dense(Affine2::new(
            filters
                .as_array()
                .to_owned()
                .mapv(|x| <f64 as num::NumCast>::from(x).unwrap()),
            bias.as_array()
                .to_owned()
                .mapv(|x| <f64 as num::NumCast>::from(x).unwrap()),
        )))
    }

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

    fn add_maxpool(&mut self, pool_size: usize) {
        self.dnn.add_layer(Layer::new_maxpool(pool_size))
    }

    fn add_flatten(&mut self) {
        self.dnn.add_layer(Layer::Flatten)
    }

    fn add_relu(&mut self, ndim: usize) {
        self.dnn.add_layer(Layer::new_relu(ndim))
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
struct PyConstellation {
    constellation: Constellation<f64, Ix2>,
}

#[pymethods]
impl PyConstellation {
    #[new]
    pub fn py_new(
        py_dnn: PyDNN,
        input_bounds: Option<(PyReadonlyArray1<f64>, PyReadonlyArray1<f64>)>,
        loc: PyReadonlyArray1<f64>,
        scale: PyReadonlyArray2<f64>,
    ) -> Self {
        let dnn = py_dnn.dnn;
        let input_shape = dnn.input_shape();
        let bounds = input_bounds.map(|(lbs, ubs)| Bounds::new(lbs.as_array(), ubs.as_array()));

        let star = match input_shape.rank() {
            1 => {
                let mut star = Star2::default(&input_shape);
                if let Some(ref b) = bounds {
                    star = star.with_input_bounds((*b).clone());
                }
                star
            }
            _ => {
                panic!()
            }
        };
        Self {
            constellation: Constellation::new(
                star,
                dnn,
                bounds,
                loc.as_array().to_owned(),
                scale.as_array().to_owned(),
            ),
        }
    }

    pub fn get_input_bounds(&self) -> Option<(Py<PyArray1<f64>>, Py<PyArray1<f64>>)> {
        let input_bounds = self
            .constellation
            .get_input_bounds()
            .as_ref()
            .map(Bounds::as_tuple);
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
        let mut star = Star2::default(&self.constellation.get_dnn().input_shape());
        if let Some(ref b) = bounds {
            star = star.with_input_bounds((*b).clone());
        }
        self.constellation.reset_with_star(star, bounds);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bounded_sample_input_multivariate_gaussian(
        &mut self,
        safe_value: f64,
        cdf_samples: usize,
        num_samples: usize,
        max_iters: usize,
        time_limit: Option<u64>,
    ) -> Option<(Py<PyArray1<f64>>, f64, f64)> {
        let mut rng = thread_rng();
        let mut asterism = Asterism::new(&mut self.constellation, safe_value);
        let output = asterism.sample_safe_star(
            num_samples,
            &mut rng,
            cdf_samples,
            max_iters,
            time_limit.map(|x| Duration::from_millis(x)),
        );
        if output.is_none() {
            return None;
        }
        let (samples, path_logp, invalid_cdf_proportion) = output.unwrap();
        let gil = Python::acquire_gil();
        let py = gil.python();
        Some((
            PyArray1::from_array(py, &samples[0]).to_owned(),
            path_logp,
            invalid_cdf_proportion,
        ))
    }
}

/// # Errors
#[pymodule]
pub fn nnv_py(_py: Python, m: &PyModule) -> PyResult<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("error")).init();
    m.add_class::<PyConstellation>()?;
    m.add_class::<PyDNN>()?;
    Ok(())
}
