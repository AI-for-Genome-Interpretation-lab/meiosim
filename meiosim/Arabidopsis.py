import h5py
import numpy as np
import pandas as pd
import os
from .Population import Population, MISSING

_pkg_dir = os.path.dirname(__file__)
_data_dir = os.path.join(_pkg_dir, "data")
_metdata_dir = os.path.join(_data_dir, "A. thaliana Master Accession List 26.01 - master_list_25.12.csv")

# Missing-call code used by the source HDF5 panel.
_SOURCE_MISSING = -1

class Arabidopsis(Population):

    def __init__(
            self,
            n_SNPs = 10_000,
            seed = 1,
        ):

        # ### REGMAP SNPs:
        # Horton et al. 2012 - Genome-wide patterns of genetic variation in worldwide Arabidopsis thaliana accessions from the RegMap panel
        # Pisupati et al. 2017 - Verification of Arabidopsis stock collections using SNPmatch, a tool for genotyping high-plexed samples
        # https://figshare.com/articles/dataset/SNP_dataset_for_A_thaliana_RegMap_panel/5514385?file=9547045
        #
        # ### 1001 Genomes:
        # The 1001 Genomes Consortium; 2016 - 1,135 Genomes Reveal the Global Pattern of Polymorphism in Arabidopsis thaliana
        # https://1001genomes.org/data/GMI-MPI/releases/v3.1/SNP_matrix_imputed_hdf5/

        SNPs_dir = os.path.join(_data_dir, "1001_genomes_filtered_on_regmap_SNPs.hdf5")

        with h5py.File(SNPs_dir, 'r') as f:

            SNPs = f['snps'][:]
            SNPs = SNPs.T
            accessions_raw = f['accessions'][:]
            accessions = pd.Series([acc.decode('utf-8') for acc in accessions_raw])
            positions = f['positions'][:]

        if accessions.duplicated().any():
            raise ValueError("Duplicated accession IDs in the panel")

        metadata = pd.read_csv(_metdata_dir)
        metadata = metadata.set_index('id').reindex(accessions.astype(int)).reset_index()
        metadata["individual"] = accessions.astype(str)

        # reduce the data set
        rng = np.random.default_rng(seed)
        max_SNPs = SNPs.shape[1]
        if n_SNPs != 0:
            indices = rng.choice(max_SNPs, size=min(n_SNPs, max_SNPs), replace=False)
            indices = np.sort(indices)
            SNPs = SNPs[:, indices]
        else:
            indices = np.arange(max_SNPs)

        # Convert to 0 / 1 / 2
        missing = SNPs == _SOURCE_MISSING
        SNPs = np.ascontiguousarray(SNPs).astype(np.int8) * 2
        SNPs[missing] = MISSING

        # genetic map
        descents = np.diff(positions) < 0
        chromosomes = np.zeros(len(positions), dtype=int)
        chromosomes[1:] = np.cumsum(descents)
        chromosomes += 1

        map = pd.DataFrame({
            'Mb': positions / 1e6,
            'chromosome': chromosomes
        })
        map = map.iloc[indices]
        
        # ### Genetic map
        # Salomé et al. 2012 - The recombination landscape in Arabidopsis thaliana F2 populations

        map["cM"] = map["Mb"] * 3.6 # cM/Mb

        super().__init__(
            genotypes = SNPs,
            metadata = metadata,
            map = map,
            seed=int(rng.integers(0, 2**63)),
        )

        # ### Phenotypes
        # Anchored on the genotyped panel: accessions absent from the SNP matrix
        # are dropped, accessions without phenotype are kept with missing values.
        #
        # Grimm et al. 2017 - easyGWAS: A Cloud-Based Platform for Comparing the
        # Results of Genome-Wide Association Studies
        # https://arapheno.1001genomes.org/study/38/

        pheno = pd.DataFrame({"accession_id": metadata["individual"].astype(int)})

        easygwas = pd.read_csv(
            os.path.join(_data_dir, "all_phenotypes_from_easyGWAS.csv")
        )
        pheno = pheno.merge(easygwas, on="accession_id", how="left")

        # ### Vernalization response
        # Flowering time after 0, 2, 4 and 8 weeks of vernalization, as accession
        # means in days from germination to first flower. Plants that had not
        # flowered by the end of the experiment sit at the upper bound (125 days)
        # and are kept on the continuous scale.
        #
        # Zhao et al. 2007 - An Arabidopsis Example of Association Mapping in
        # Structured Samples, PLoS Genetics 3(1): e4
        # Redistributed in the 107-phenotype compilation of Atwell et al. 2010
        # https://arapheno.1001genomes.org/study/12/

        vernalization = pd.read_csv(
            os.path.join(_data_dir, "vernalization_flowering_time.csv")
        )
        pheno = pheno.merge(vernalization, on="accession_id", how="left")

        pheno["individual"] = pheno["accession_id"].astype(str)

        self.phenotypes = pheno


        



