import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import hashlib
from concurrent.futures import ProcessPoolExecutor


def _meiosis_chromosome_task(task):
    """One task = one chromosome of one gamete of one progeny.

    Returns its reassembly address (desc_idx, gamete_idx, indices) along
    with the produced haplotype, so it can be placed back without relying
    on the pool's return order.
    """
    rng = np.random.default_rng(task['seed'])
    cM = task['cM']
    hap0, hap1 = task['phases_chrom']
    n = len(cM)

    # RNG consumption order identical to the loop version:
    #   1 integers(0,2) then (n-1) uniform draws.
    # rng.random(n-1) == n-1 successive rng.random() calls -> bit-for-bit repro.
    start = rng.integers(0, 2)
    distances_cM = np.diff(cM)                       # transitions to i = 1..n-1
    switches = rng.random(n - 1) < distances_cM / 100

    # current[i] = (start + number of crossovers up to i) % 2
    current = np.empty(n, dtype=int)
    current[0] = start
    current[1:] = (start + np.cumsum(switches)) % 2

    hap = np.where(current == 0, hap0, hap1)

    return task['desc_idx'], task['gamete_idx'], task['chrom_indices'], hap

class Population():

    @classmethod
    def merge(cls, *populations):
        
        if len(populations) < 2:
            raise ValueError("At least 2 populations are required")
        
        # Map
        reference_map = populations[0].map
        for _, pop in enumerate(populations[1:], start=1):
            if not reference_map.equals(pop.map):
                raise ValueError(f"All maps are not identical")
        
        # Metadata
        all_metadata = pd.concat(
            [pop.metadata for pop in populations],
            axis=0,
            ignore_index=True,
            sort=False,
        )
        
        # Genotypes
        all_genotypes = np.vstack([pop.genotypes for pop in populations])
        
        # Output
        merged = cls(
            genotypes=all_genotypes,
            metadata=all_metadata,
            map=reference_map.copy(),
            seed=int(populations[0].rng.integers(0, 2**63)),
        )
        
        # Phases if they exist
        if all(hasattr(pop, 'phases') and pop.phases is not None for pop in populations):
            merged.phases = [
                np.vstack([pop.phases[0] for pop in populations]),
                np.vstack([pop.phases[1] for pop in populations])
            ]
        
        return merged

    def __init__(self, genotypes, metadata, map, seed=None):
        self.genotypes = genotypes
        self.metadata = metadata
        self.map = map
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if 'individual' not in self.metadata.columns:
            self.metadata['individual'] = [
                hashlib.sha256(row.tobytes()).hexdigest()[:16]
                for row in self.genotypes
            ]
            
        for parent in ["sire","dam"]:
            if parent not in self.metadata.columns:
                self.metadata[parent] = ""

    def subset(self, on: dict | list[int | str]):

        if isinstance(on, dict):
            mask = np.ones(len(self.metadata), dtype=bool)
        
            for column, values in on.items():
                if column not in self.metadata.columns:
                    raise ValueError(f"Column '{column}' not found in metadata")

                column_mask = np.zeros(len(self.metadata), dtype=bool)

                if isinstance(values, str):
                    values = [values]

                for value in values:
                    column_mask |= (self.metadata[column] == value)

                mask &= column_mask

        elif isinstance(on, list):
            if len(on) == 0:
                mask = np.zeros(len(self.metadata), dtype=bool)
            
            elif all(isinstance(item, str) for item in on):
                mask = self.metadata['individual'].isin(on).values
            
            elif all(isinstance(item, (int, np.integer)) for item in on):
                indices = np.array(on)
                if np.any(indices >= len(self.metadata)) or np.any(indices < 0):
                    raise ValueError("Indexes are out of scope")
                mask = np.zeros(len(self.metadata), dtype=bool)
                mask[indices] = True
            
        else:
            raise TypeError("list items must be all str or all int")
            
        self.metadata = self.metadata[mask].reset_index(drop=True)
        self.genotypes = self.genotypes[mask, :]

        if hasattr(self, 'phases') and self.phases is not None:
            self.phases = [
                self.phases[0][mask],
                self.phases[1][mask]
            ]

        if getattr(self, "phenotypes", None) is not None:
            self.phenotypes = self.phenotypes[
                self.phenotypes["individual"].isin(self.metadata["individual"])
            ].reset_index(drop=True)
                          
        return self

    def split(self, size: int):
        
        total_snps = self.genotypes.shape[1]
        
        if size <= 0 or size >= total_snps:
            raise ValueError(f"size must be between 1 and {total_snps - 1}")
        indices1 = self.rng.choice(total_snps, size=size, replace=False)
        
        mask1 = np.zeros(total_snps, dtype=bool)
        mask1[indices1] = True
        mask2 = ~mask1
        
        pop1 = Population(
            genotypes=self.genotypes[:, mask1],
            metadata=self.metadata.copy(),
            map=self.map[mask1].reset_index(drop=True),
            seed=int(self.rng.integers(0, 2**63)),
        )

        pop2 = Population(
            genotypes=self.genotypes[:, mask2],
            metadata=self.metadata.copy(),
            map=self.map[mask2].reset_index(drop=True),
            seed=int(self.rng.integers(0, 2**63)),
        )

        if hasattr(self, 'phases') and self.phases is not None:
            pop1.phases = [
                self.phases[0][:, mask1],
                self.phases[1][:, mask1]
            ]
            pop2.phases = [
                self.phases[0][:, mask2],
                self.phases[1][:, mask2]
            ]

        return pop1, pop2

    def plot(self, group = None):

        metadata = self.metadata.copy()
        genotypes = self.genotypes.copy()

        imputer = SimpleImputer(strategy='mean')
        genotypes_imputed = imputer.fit_transform(genotypes)

        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(genotypes_imputed)

        ax = plt.gca()

        if group is not None:
            pca_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
                'group': metadata[group].values,
            })

            groups = sorted(pca_df['group'].unique())
            n_groups = len(groups)

            if 'other' in groups:
                groups.remove('other')
                palette = sns.color_palette('husl', len(groups))
                color_map = dict(zip(groups, palette))
                color_map['other'] = (0.5, 0.5, 0.5)
                groups.append('other')
            else:
                palette = sns.color_palette('husl', n_groups)
                color_map = dict(zip(groups, palette))

            for country in groups:
                mask = pca_df['group'] == country
                ax.scatter(
                    pca_df.loc[mask, 'PC1'],
                    pca_df.loc[mask, 'PC2'],
                    label=country,
                    alpha=0.8,
                    s=60,
                    edgecolors='white',
                    linewidth=0.5,
                    color=color_map[country]
                )

            legend = ax.legend(
                title= group,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                frameon=True,
                fancybox=True,
                shadow=True,
                ncol=1 if n_groups <= 15 else 2
            )
            legend.get_title().set_fontsize(12)
            legend.get_title().set_fontweight('bold')

        else:
            pca_df = pd.DataFrame({
                'PC1': pca_result[:, 0],
                'PC2': pca_result[:, 1],
            })
            ax.scatter(
                pca_df.loc[:, 'PC1'],
                pca_df.loc[:, 'PC2'],
                alpha=0.8,
                s=60,
                edgecolors='white',
                linewidth=0.5,
            )

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)', fontsize=14, fontweight='bold')

        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.5)

    def phasing(self):
        genotypes = self.genotypes.copy()
        imputer = SimpleImputer(strategy='most_frequent')
        g = imputer.fit_transform(genotypes)
        haplotypes = g / 2          # -1 -> -0.5, 0 -> 0, +1 -> +0.5
        self.phases = [haplotypes, haplotypes]

    def _chromosome_layout(self):
        """Precompute, once, (indices, cM) per chromosome."""
        layout = []
        for chrom in self.map['chromosome'].unique():
            idx = np.where((self.map['chromosome'] == chrom).values)[0]
            cM = self.map['cM'].values[idx]
            layout.append((chrom, idx, cM))
        return layout

    def cross(
            self,
            parents: list[str | int] | str | int = None,
            selfing: bool = True,
            n_progeny: int = 1,
            n_cores: int = 5,
        ):

        if not hasattr(self, 'phases') or self.phases is None:
            self.phasing()

        # --- resolve parents ---
        if parents is None:
            parents = list(range(len(self.metadata)))
        else:
            if not isinstance(parents, list):
                parents = [parents]

            id_to_idx = {id_val: idx for idx, id_val in enumerate(self.metadata["individual"])}
            normalized = []
            for p in parents:
                if isinstance(p, int):
                    normalized.append(p)
                elif isinstance(p, str):
                    normalized.append(id_to_idx[p])
            parents = normalized

        # --- build the list of crosses, then of progeny ---
        if selfing:
            crosses = [(p1, p2) for p1 in parents for p2 in parents if p1 <= p2]
        else:
            crosses = [(p1, p2) for p1 in parents for p2 in parents if p1 < p2]

        matings = []  # one (parent1, parent2) per progeny
        for (p1, p2) in crosses:
            for _ in range(n_progeny):
                matings.append((p1, p2))
        n_desc = len(matings)

        # --- build the FLAT task list, deterministic order ---
        # fixed order (reproducibility contract): progeny, then gamete, then chromosome
        layout = self._chromosome_layout()
        tasks = []
        for desc_idx, (p1, p2) in enumerate(matings):
            for gamete_idx, parent_idx in enumerate((p1, p2)):
                for (_chrom, idx, cM) in layout:
                    tasks.append({
                        'desc_idx': desc_idx,
                        'gamete_idx': gamete_idx,
                        'chrom_indices': idx,
                        'cM': cM,
                        # only pickle the chromosome slice, not the whole genome
                        'phases_chrom': (
                            self.phases[0][parent_idx, idx],
                            self.phases[1][parent_idx, idx],
                        ),
                        # 'seed' filled in just after
                    })

        # --- independent seeds, one per task, deterministic ---
        base = int(self.rng.integers(0, 2**63))
        child_seeds = np.random.SeedSequence(base).spawn(len(tasks))
        for task, s in zip(tasks, child_seeds):
            task['seed'] = s

        # --- a SINGLE pool, flat list distributed over n_cores ---
        n_snps = len(self.map)
        gametes = np.zeros((n_desc, 2, n_snps), dtype=float)
        chunksize = max(1, len(tasks) // (n_cores * 4))

        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            for desc_idx, gamete_idx, idx, hap in executor.map(
                _meiosis_chromosome_task, tasks, chunksize=chunksize
            ):
                gametes[desc_idx, gamete_idx, idx] = hap

        # --- reassembly ---
        all_genotypes = gametes[:, 0, :] + gametes[:, 1, :]

        pedigree = []
        for desc_idx, (p1, p2) in enumerate(matings):
            genotype = all_genotypes[desc_idx]
            pedigree.append({
                'individual': hashlib.sha256(genotype.tobytes()).hexdigest()[:16],
                'sire': self.metadata.iloc[p1]["individual"],
                'dam': self.metadata.iloc[p2]["individual"],
            })

        progeny = Population(
            genotypes=all_genotypes,
            metadata=pd.DataFrame(pedigree),
            map=self.map,
            seed=int(self.rng.integers(0, 2**63)),
        )
        progeny.phases = [gametes[:, 0, :], gametes[:, 1, :]]

        return progeny

    def selfing(
            self,
            individual: str,
            n_generations: int,
            n_cores: int = 5,
        ):

        ind = self.cross(individual, selfing=True, n_cores=n_cores)

        for _ in range(1, n_generations):
            ind = ind.cross(0, selfing=True, n_cores=n_cores)

        source_metadata = self.metadata.loc[self.metadata['individual'] == individual].iloc[0]
        ind.metadata['sire'] = source_metadata.get('sire', None)
        ind.metadata['dam'] = source_metadata.get('dam', None)

        return ind
    
