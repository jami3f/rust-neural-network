use rand::{Rng, RngCore};

pub struct LayerTopology {
    pub neurons: usize,
}

#[derive(Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .windows(2)
            .map(|layers| Layer::random(rng, layers[0].neurons, layers[1].neurons))
            .collect();

        Self { layers }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(inputs))
    }
}
#[derive(Debug, Clone)]
struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(neurons: Vec<Neuron>) -> Self {
        Self { neurons }
    }
    pub fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }

    fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}
#[derive(Debug, Clone)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    pub fn new(bias: f32, weights: Vec<f32>) -> Self {
        Self { bias, weights }
    }

    pub fn random(rng: &mut dyn rand::RngCore, output_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (output + self.bias).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx;
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    mod neuron {
        use super::*;
        mod random {
            use super::*;

            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());
                let neuron = Neuron::random(&mut rng, 4);

                assert_relative_eq!(neuron.bias, -0.6255188);
                assert_relative_eq!(
                    neuron.weights.as_slice(),
                    [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref()
                );
            }
        }

        mod propagate {
            use super::*;

            #[test]
            fn test() {
                let neuron = Neuron {
                    bias: 0.5,
                    weights: vec![-0.3, 0.8],
                };

                assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0);
                assert_relative_eq!(
                    neuron.propagate(&[0.5, 1.0]),
                    (-0.3 * 0.5) + (0.8 * 1.0) + 0.5
                );
            }
        }
    }
    mod layer {
        use super::*;
        mod random {
            use super::*;
            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());
                let layer = Layer::random(&mut rng, 3, 2);
                let expected_biases = vec![-0.6255188, 0.5238807];
                let actual_biases: Vec<f32> =
                    layer.neurons.iter().map(|neuron| neuron.bias).collect();

                let expected_weights: Vec<&[f32]> = vec![
                    &[0.67383957, 0.8181262, 0.26284897],
                    &[-0.53516835, 0.069369674, -0.7648182],
                ];
                let actual_weights: Vec<&[f32]> = layer
                    .neurons
                    .iter()
                    .map(|neuron| neuron.weights.as_slice())
                    .collect();

                assert_relative_eq!(expected_biases.as_slice(), actual_biases.as_slice());
                assert_relative_eq!(expected_weights.as_slice(), actual_weights.as_slice());
            }
        }
        mod propagate {
            use super::*;
            #[test]
            fn test() {
                let neurons = &[
                    Neuron::new(0.0, vec![0.1, 0.2, 0.3]),
                    Neuron::new(0.0, vec![0.4, 0.5, 0.6]),
                ];
                let layer = Layer::new(vec![neurons[0].clone(), neurons[1].clone()]);

                let inputs: &[f32] = &[-0.5, 0.0, 0.5];
                let expected = vec![neurons[0].propagate(inputs), neurons[1].propagate(inputs)];
                let actual = layer.propagate(inputs.to_vec());

                assert_relative_eq!(expected.as_slice(), actual.as_slice());
            }
        }
    }
    mod network {
        use super::*;
        mod random {
            use super::*;
            #[test]
            fn test() {
                let mut rng = ChaCha8Rng::from_seed(Default::default());

                let layer_topologies = &[
                    LayerTopology { neurons: 3 },
                    LayerTopology { neurons: 2 },
                    LayerTopology { neurons: 1 },
                ];

                let network = Network::random(&mut rng, layer_topologies);

                let expected_biases: Vec<Vec<f32>> =
                    vec![vec![-0.6255188, 0.5238807], vec![-0.102499366]];
                let actual_biases: Vec<Vec<f32>> = network
                    .layers
                    .iter()
                    .map(|layer| layer.neurons.iter().map(|neuron| neuron.bias).collect())
                    .collect();

                for i in 0..expected_biases.len() {
                    assert_relative_eq!(expected_biases[i].as_slice(), actual_biases[i].as_slice());
                }

                let expected_weights = vec![
                    vec![
                        vec![0.67383957, 0.8181262, 0.26284897],
                        vec![-0.53516835, 0.069369674, -0.7648182],
                    ],
                    vec![vec![-0.48879617, -0.19277132]],
                ];

                let actual_weights: Vec<Vec<Vec<f32>>> = network
                    .layers
                    .iter()
                    .map(|layer| {
                        layer
                            .neurons
                            .iter()
                            .map(|neuron| neuron.weights.clone())
                            .collect()
                    })
                    .collect();

                for i in 0..expected_weights.len() {
                    for j in 0..expected_weights[i].len() {
                        assert_relative_eq!(
                            expected_weights[i][j].as_slice(),
                            actual_weights[i][j].as_slice()
                        );
                    }
                }
            }
        }
        mod propagate {
            #[test]
            fn test() {
                todo!()
            }
        }
    }
}
