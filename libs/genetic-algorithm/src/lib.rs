#![feature(impl_trait_in_assoc_type)]

use std::ops::Index;

use rand::{seq::SliceRandom, RngCore};

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
}

pub trait SelectionMethod {
    fn select<'a, I>(&self, rng: &mut dyn RngCore, population: &'a [I]) -> &'a I
    where
        I: Individual;
}

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

pub struct GeneticAlgorithm<S> {
    selection_method: S,
}

impl<S> GeneticAlgorithm<S>
where
    S: SelectionMethod,
{
    pub fn new(selection_method: S) -> Self {
        Self { selection_method }
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
                todo!()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    mod genetic_algorithm {
        use super::*;
        use maplit;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;
        use core::panic;
        use std::collections::BTreeMap;

        #[cfg(test)]
        pub struct TestIndividual {
            fitness: f32,
        }

        impl TestIndividual {
            pub fn new(fitness: f32) -> Self {
                Self { fitness }
            }
        }

        impl Individual for TestIndividual {
            fn fitness(&self) -> f32 {
                self.fitness
            }

            fn chromosome(&self) -> &Chromosome {
                panic!("Not supported for TestIndividual")
            }
        }

        #[test]
        fn test() {
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
                let genes:Vec<_> = chromosome.into_iter().collect();

                assert_eq!(genes[0], 3.0);
                assert_eq!(genes[1], 1.0);
                assert_eq!(genes[2], 2.0);
            }
        }
    }
}
