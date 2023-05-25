#![feature(impl_trait_in_assoc_type)]

use std::ops::Index;

use rand::{seq::SliceRandom, Rng, RngCore};

pub trait MutationMethod {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome);
}

pub struct GaussianMutation {
    chance: f32,
    coeff: f32,
}

impl GaussianMutation {
    fn new(chance: f32, coeff: f32) -> Self {
        assert!(chance >= 0.0 && chance <= 1.0);

        Self { chance, coeff }
    }
}

impl MutationMethod for GaussianMutation {
    fn mutate(&self, rng: &mut dyn RngCore, child: &mut Chromosome) {
        for gene in child.iter_mut() {
            let sign = if rng.gen_bool(0.5) { -1.0 } else { 1.0 };

            if rng.gen_bool(self.chance as _) {
                *gene += sign * self.coeff * rng.gen::<f32>();
            }
        }
    }
}

pub trait CrossoverMethod {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome;
}

pub struct UniformCrossover;

impl UniformCrossover {
    pub fn new() -> Self {
        Self
    }
}

impl CrossoverMethod for UniformCrossover {
    fn crossover(
        &self,
        rng: &mut dyn RngCore,
        parent_a: &Chromosome,
        parent_b: &Chromosome,
    ) -> Chromosome {
        assert_eq!(parent_a.len(), parent_b.len());

        let parent_a = parent_a.iter();
        let parent_b = parent_b.iter();

        parent_a
            .zip(parent_b)
            .map(|(&a, &b)| if rng.gen_bool(0.5) { a } else { b })
            .collect()
    }
}

pub struct RouletteWheelSelection;

impl RouletteWheelSelection {
    pub fn new() -> Self {
        Self
    }
}

impl SelectionMethod for RouletteWheelSelection {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual,
    {
        population
            .choose_weighted(rng, |individual| individual.fitness())
            .expect("got an empty population")
    }
}

pub trait Individual {
    fn fitness(&self) -> f32;
    fn chromosome(&self) -> &Chromosome;
    fn create(chromosome: Chromosome) -> Self;
}

pub trait SelectionMethod {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual;
}
#[derive(Debug)]
pub struct Chromosome {
    genes: Vec<f32>,
}

impl Chromosome {
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = &f32> {
        self.genes.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.genes.iter_mut()
    }
}

impl Index<usize> for Chromosome {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.genes[index]
    }
}

impl FromIterator<f32> for Chromosome {
    fn from_iter<T: IntoIterator<Item = f32>>(iter: T) -> Self {
        Self {
            genes: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for Chromosome {
    type Item = f32;
    type IntoIter = impl Iterator<Item = f32>;

    fn into_iter(self) -> Self::IntoIter {
        self.genes.into_iter()
    }
}

#[cfg(test)]
impl PartialEq for Chromosome {
    fn eq(&self, other: &Self) -> bool {
        approx::relative_eq!(self.genes.as_slice(), other.genes.as_slice())
    }
}

pub struct GeneticAlgorithm<S> {
    selection_method: S,
    crossover_method: Box<dyn CrossoverMethod>,
    mutation_method: Box<dyn MutationMethod>,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(
        selection_method: S,
        crossover_method: impl CrossoverMethod + 'static,
        mutation_method: impl MutationMethod + 'static,
    ) -> Self {
        Self {
            selection_method,
            crossover_method: Box::new(crossover_method),
            mutation_method: Box::new(mutation_method),
        }
    }

    pub fn evolve<I>(&self, rng: &mut dyn RngCore, population: &[I]) -> Vec<I>
    where
        I: Individual,
    {
        assert!(!population.is_empty());
        (0..population.len())
            .map(|_| {
                let parent_a = self.selection_method.select(rng, population).chromosome();
                let parent_b = self.selection_method.select(rng, population).chromosome();

                let mut child = self.crossover_method.crossover(rng, parent_a, parent_b);

                self.mutation_method.mutate(rng, &mut child);

                I::create(child)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod genetic_algorithm {
        use super::*;
        use core::panic;
        use maplit;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use std::{collections::BTreeMap, fmt::format};

        #[cfg(test)]
        #[derive(Debug, PartialEq)]
        pub enum TestIndividual {
            WithCromosome { chromosome: Chromosome },
            WithFitness { fitness: f32 },
        }

        impl TestIndividual {
            pub fn new(fitness: f32) -> Self {
                Self::WithFitness { fitness }
            }
        }

        impl Individual for TestIndividual {
            fn create(chromosome: Chromosome) -> Self {
                Self::WithCromosome { chromosome }
            }

            fn chromosome(&self) -> &Chromosome {
                match self {
                    Self::WithCromosome { chromosome } => chromosome,

                    Self::WithFitness { fitness } => {
                        panic!("Not supported for TestIndividual::WithFitness")
                    }
                }
            }

            fn fitness(&self) -> f32 {
                match self {
                    Self::WithCromosome { chromosome } => chromosome.iter().sum(),

                    Self::WithFitness { fitness } => *fitness,
                }
            }
        }

        fn individual(genes: &[f32]) -> TestIndividual {
            let chromosome = genes.iter().cloned().collect();

            TestIndividual::create(chromosome)
        }

        #[test]
        fn test_selection() {
            let method = RouletteWheelSelection::new();
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let population = vec![
                TestIndividual::new(2.0),
                TestIndividual::new(1.0),
                TestIndividual::new(4.0),
                TestIndividual::new(3.0),
            ];

            let expected_histogram = maplit::btreemap! {1 => 98, 2 => 202, 3 => 278, 4 => 422};

            let actual_histogram: BTreeMap<i32, _> = (0..1000)
                .map(|_| method.select(&mut rng, &population))
                .fold(Default::default(), |mut histogram, individual| {
                    *histogram.entry(individual.fitness() as _).or_default() += 1;
                    histogram
                });

            assert_eq!(expected_histogram, actual_histogram);
        }

        #[test]
        fn test_genetic_algorithm() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());

            let ga = GeneticAlgorithm::new(
                RouletteWheelSelection::new(),
                UniformCrossover::new(),
                GaussianMutation::new(0.5, 0.5),
            );

            let mut population = vec![
                individual(&[0.0, 0.0, 0.0]),
                individual(&[1.0, 1.0, 1.0]),
                individual(&[1.0, 2.0, 1.0]),
                individual(&[1.0, 2.0, 4.0]),
            ];

            for _ in 0..10 {
                population = ga.evolve(&mut rng, &population);
            }

            let expected_population = vec![
                individual(&[0.4476949, 2.0648358, 4.3058133]),
                individual(&[1.2126867, 1.5538777, 2.886911]),
                individual(&[1.0617678, 2.265739, 4.428764]),
                individual(&[0.95909685, 2.4618788, 4.024733]),
            ];

            assert_eq!(population, expected_population);
        }
    }

    mod chromosome {
        use super::*;

        fn chromosome() -> Chromosome {
            Chromosome {
                genes: vec![3.0, 1.0, 2.0],
            }
        }

        mod len {
            use super::*;

            #[test]
            fn test() {
                assert_eq!(chromosome().len(), 3);
            }
        }

        mod iter {
            use super::*;

            #[test]
            fn test() {
                let chromosome = chromosome();
                let genes: Vec<_> = chromosome.iter().collect();

                assert_eq!(genes.len(), 3);
                assert_eq!(genes[0], &3.0);
                assert_eq!(genes[1], &1.0);
                assert_eq!(genes[2], &2.0);
            }
        }

        mod iter_mut {
            use super::*;

            #[test]
            fn test() {
                let mut chromosome = chromosome();
                chromosome.iter_mut().for_each(|gene| {
                    *gene *= 10.0;
                });

                let genes: Vec<_> = chromosome.iter().collect();

                assert_eq!(genes.len(), 3);
                assert_eq!(genes[0], &30.0);
                assert_eq!(genes[1], &10.0);
                assert_eq!(genes[2], &20.0);
            }
        }

        mod index {
            use super::*;

            #[test]
            fn test() {
                let chromosome = chromosome();

                assert_eq!(chromosome[0], 3.0);
                assert_eq!(chromosome[1], 1.0);
                assert_eq!(chromosome[2], 2.0);
            }
        }

        mod from_iterator {
            use super::*;

            #[test]
            fn test() {
                let chromosome: Chromosome = vec![3.0, 1.0, 2.0].into_iter().collect();

                assert_eq!(chromosome[0], 3.0);
                assert_eq!(chromosome[1], 1.0);
                assert_eq!(chromosome[2], 2.0);
            }
        }

        mod into_iterator {
            use super::*;

            #[test]
            fn test() {
                let chromosome: Chromosome = chromosome();
                let genes: Vec<_> = chromosome.into_iter().collect();

                assert_eq!(genes[0], 3.0);
                assert_eq!(genes[1], 1.0);
                assert_eq!(genes[2], 2.0);
            }
        }
    }

    mod uniform_crossover {
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        use super::*;

        #[test]
        fn test() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let parent_a: Chromosome = (1..=100).map(|n| n as f32).collect();
            let parent_b: Chromosome = (1..=100).map(|n| -n as f32).collect();

            let child = UniformCrossover::new().crossover(&mut rng, &parent_a, &parent_b);

            let diff_a = child.iter().zip(parent_a).filter(|(c, p)| *c != p).count();
            let diff_b = child.iter().zip(parent_b).filter(|(c, p)| *c != p).count();

            assert_eq!(diff_a, 49);
            assert_eq!(diff_b, 51);
        }
    }

    mod gaussian_mutation {
        use approx::assert_relative_eq;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        use super::*;

        fn actual(chance: f32, coeff: f32) -> Vec<f32> {
            let mut child = vec![1.0, 2.0, 3.0, 4.0, 5.0].into_iter().collect();

            let mut rng = ChaCha8Rng::from_seed(Default::default());

            GaussianMutation::new(chance, coeff).mutate(&mut rng, &mut child);

            child.into_iter().collect()
        }

        mod given_zero_chance {
            use super::*;

            fn actual(coeff: f32) -> Vec<f32> {
                super::actual(0.0, coeff)
            }
            mod and_zero_coefficient {
                use super::*;
                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_coefficient {
                use super::*;
                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }

        mod given_fifty_fifty_chance {
            use super::*;

            fn actual(coeff: f32) -> Vec<f32> {
                super::actual(0.5, coeff)
            }
            mod and_zero_coefficient {
                use super::*;
                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_coefficient {
                use super::*;
                #[test]
                fn slightly_changes_the_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.0, 1.7756249, 3.0, 4.1596804, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }

        mod given_max_chance {
            use super::*;

            fn actual(coeff: f32) -> Vec<f32> {
                super::actual(1.0, coeff)
            }
            mod and_zero_coefficient {
                use super::*;
                #[test]
                fn does_not_change_the_original_chromosome() {
                    let actual = actual(0.0);
                    let expected = vec![1.0, 2.0, 3.0, 4.0, 5.0];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }

            mod and_nonzero_coefficient {
                use super::*;
                #[test]
                fn entirely_changes_the_original_chromosome() {
                    let actual = actual(0.5);
                    let expected = vec![1.4545316, 2.1162078, 2.7756248, 3.9505124, 4.638691];

                    assert_relative_eq!(actual.as_slice(), expected.as_slice());
                }
            }
        }
    }
}
