"""
Sensitivity addendum unit tests (protocol/sensitivity_addendum_v1.md):
  1. Exact-key isolation: for every (protocol, instance), no test molecule's
     exact canonical SMILES appears in the training partition (global
     allocation assertion over the frozen split construction).
  2. Level-1 standardized-parent key determinism.
Run: python3 tests/test_identity_isolation.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
from b3_config import SEED_BASE
from b3_main_v2 import global_split3

HERE = os.path.dirname(os.path.abspath(__file__))
IDEA = os.path.join(HERE, '..', '..', '..', 'idea-stage')
sys.path.insert(0, IDEA)
from b3_main_experiment import load_endpoints


def test_exact_key_isolation():
    ep_data = load_endpoints()
    all_mols = set()
    for ep, df in ep_data.items():
        all_mols |= set(df['canon'])
    failures = []
    for protocol in ['random', 'scaffold']:
        for inst in range(5):
            train_set, cal_set, test_set = global_split3(
                all_mols, seed=SEED_BASE + inst, scaffold=(protocol == 'scaffold'))
            if test_set & train_set:
                failures.append((protocol, inst, len(test_set & train_set)))
            if test_set & cal_set:
                failures.append((protocol, inst, 'cal-collision'))
            # per-endpoint: no test molecule appears in any endpoint's training rows
            for ep, df in ep_data.items():
                te_mols = set(df['canon']) & test_set
                if te_mols & train_set:
                    failures.append((protocol, inst, ep))
    assert not failures, f'exact-key isolation violated: {failures}'
    print('PASS: exact canonical-SMILES isolation holds for all protocols x instances '
          '(no test molecule in any training partition)')


def test_l1_determinism():
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as rms
    _te = rms.TautomerEnumerator()
    _unch = rms.Uncharger()

    def key(smi):
        mol = Chem.MolFromSmiles(smi)
        mol = rms.Cleanup(mol)
        mol = rms.FragmentParent(mol)
        mol = _unch.uncharge(mol)
        mol = _te.Canonicalize(mol)
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    samples = ['CCO', 'c1ccccc1C(=O)O', 'CC(=O)Oc1ccccc1C(=O)O',
               'N[C@@H](C)C(=O)O', 'c1nccc2[nH]ccc12']
    for smi in samples:
        a, b = key(smi), key(smi)
        assert a == b, f'non-deterministic L1 key for {smi}'
    print('PASS: L1 standardized-parent keys are deterministic')


if __name__ == '__main__':
    test_exact_key_isolation()
    test_l1_determinism()
    print('all identity isolation tests passed')
