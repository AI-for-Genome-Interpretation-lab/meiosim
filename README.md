# meiosim

**meiosim** provides a single, lightweight data structure that bundles genotypes, individual metadata and a genetic map together. It includes the operations breeders and quantitative geneticists reach for most often:

- basic data manipulation,
- simulating meiosis and crosses,
- a ready-to-use *Arabidopsis thaliana* panel.

<div align="center">
  <img src="https://github.com/user-attachments/assets/8825fe6b-a844-410b-9ab2-306129f83fc9" alt="Arabidopsis panel" width="750">
</div>

## Installation

Install meiosim with pip:

```bash
pip install meiosim
```

## Quick start

```python
from meiosim import Arabidopsis
import matplotlib.pyplot as plt

# Load 5,000 random SNPs from the 1001 Genomes panel
pop = Arabidopsis(n_SNPs = 5000, seed = 42)

# Visualise population structure
plt.figure(figsize=(12,6))
pop.plot(group="country")
plt.show()
```

## Core concepts

Everything in **meiosim** revolves around the `Population` object, which holds three
aligned pieces of data:

| Attribute    | Type            | Shape                          | Description                                                            |
|--------------|-----------------|--------------------------------|------------------------------------------------------------------------|
| `genotypes`  | `np.ndarray`    | `(n_individuals, n_markers)`   | Allele dosage coded `0 / 1 / 2`; missing values are `-128`.          |
| `metadata`   | `pd.DataFrame`  | `n_individuals` rows           | One row per individual. Columns are free-form.                   |
| `map`        | `pd.DataFrame`  | `n_markers` rows               | Marker map; needs at least `chromosome` and `cM` columns.              |

On construction, **meiosim** guarantees three metadata columns:

- **`individual`**: a primary key for each individual, automatically created as a 16-character SHA-256 fingerprint if absent.
- **`sire`** / **`dam`**: parental identifiers, filled in by crossing operations.

Optional attributes appear once you run the relevant method:

- **`phases`**: a list `[hap0, hap1]` of two haplotype matrices, created by
  `phasing()` and propagated through crossing.
- **`phenotypes`**: a `pd.DataFrame` of trait values.

A shared `numpy` random generator (`rng`) makes every stochastic operation reproducible.

## The `Population` API

### Constructor

```python
pop = Population(genotypes, metadata, map, seed = 42)
```

Build a population directly from SNP data. `genotypes` is a `0/1/2` matrix
(`np.nan` allowed), `metadata` and `map` are DataFrames aligned to the rows and
columns of `genotypes` respectively.

### `merge(*populations)` *(classmethod)*

```python
combined = Population.merge(pop_a, pop_b, pop_c)
```

Stack two or more populations that share an identical map. Metadata is
concatenated, genotypes are row-stacked, and phases are merged when all input populations carry them.

### `subset(on)`

Filter individuals in place, on either:

- **`dict`**: match metadata columns, e.g. `{"country": "SWE"}` or
  `{"country": ["SWE", "FIN"], "stage": "line"}`. Multiple keys are combined
  with `and`; multiple values per key with `or`.
- **`list[str]`**: keep individuals by their `individual` ID.
- **`list[int]`**: keep individuals by row index.

```python
pop.subset({"country": "SWE"}).subset(list(range(20)))
```

### `split(size)`

Randomly partition the markers into two new populations of `size` and
`n_markers - size` columns:

```python
snp, qtl = pop.split(2_000)   # snp has 2,000 markers, qtl gets the rest
```

Each individual is preserved in both halves; phases are split alongside.

### `plot(group=None)`

PCA scatter of the population on the first two principal components (genotypes
are mean-imputed first). Pass a metadata column name to `group` to colour points
by category (a group named `"other"` is drawn in grey).

```python
pop.plot(group="country")
```

### `phasing()`

Infer haplotypes and populate `self.phases`.

⚠️ Designed for fully inbred
material: for an actual phasing of heterozygous material, an external tool must be used.

### `cross(parents=None, selfing=True, n_progeny=1, n_cores=5)`

Generate progeny from all pairwise combinations of the chosen parents.

- `parents` — a list of IDs (`str`) or indices (`int`), or a single one.
  Defaults to every individual in the population.
- `selfing` — if `True`, self-crosses are allowed (pairs with `p1 <= p2`);
  if `False`, only distinct pairs are crossed (`p1 < p2`).
- `n_progeny` — offspring produced per cross.
- `n_cores` — workers used by the underlying meiosis.

Returns a new `Population` whose `metadata` is a pedigree (`individual`, `sire`,
`dam`) and whose `phases` carry the two inherited gametes.

```python
f1 = founders.cross(selfing=False)      # all distinct pairwise crosses
```

### `selfing(individual, n_generations, n_cores=5)`

Repeatedly self-fertilise one individual to drive it towards homozygosity,
returning the population after `n_generations` rounds. The original parents are
carried over into the resulting metadata.

```python
line = f1.selfing(id1, n_generations=6)
```

## Multiprocessing

`cross`, `selfing` and `meiosis` use a
  `ProcessPoolExecutor`. On macOS/Windows, set the start method to `'fork'`
  (where supported) before launching workers, as shown in the example, and run
  inside an `if __name__ == "__main__":` guard.

## Arabidopsis dataset

The 1001 genome collection of *A. thaliana* (The 1001 Genomes Consortium, 2016) filtered on the RegMap panel (Horton et al. 2012, Pisupati et al. 2017) is provided as a resource.

The genetic map is built from the physical map assuming a constant 3.6 cM/Mb, simplifying the recombination landscape described by Salomé et al. (2012).

Phenotypes come from Grimm et al. (2017).

The **`Arabidopsis`** constructor base loader reads SNPs, accessions, positions, derives the map and provides phenotypes. Optionally, down-samples to `n_SNPs` random markers (`n_SNPs=0` keeps them all).

| Feature | Reference                                             |
|---------|-------------------------------------------------------|
| 1001 Arabidopsis genomes | The 1001 Genomes Consortium (2016). *1,135 Genomes Reveal the Global Pattern of Polymorphism in Arabidopsis thaliana.* Cell. |
| 1001 Arabidopsis phenotypes | Grimm et al. (2017). *easyGWAS: A Cloud-Based Platform for Comparing the Results of Genome-Wide Association Studies.* The Plant Cell. |
| Genetic map | Salomé et al. (2012). *The recombination landscape in Arabidopsis thaliana F2 populations.* Heredity. |
| RegMap SNP panel | Horton et al. (2012). *Genome-wide patterns of genetic variation in worldwide Arabidopsis thaliana accessions from the RegMap panel.* Nature Genetics. |
| RegMap SNP panel | Pisupati et al. (2017) - Verification of Arabidopsis stock collections using SNPmatch, a tool for genotyping high-plexed samples |

## Citation

Please cite this package as:

Marchal, A., & Raimondi, D. (2026). meiosim - a toolbox to simulate crosses and play with quantitative genetics in Python  [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21134827

## Code

The code is available at [this address](https://github.com/ai-for-genome-interpretation-lab/meiosim).

## License

Copyright © 2026 CNRS, University of Montpellier

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program. If not, see <https://www.gnu.org/licenses/>. 
