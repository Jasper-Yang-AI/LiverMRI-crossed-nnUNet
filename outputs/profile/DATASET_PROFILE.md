# Dataset Profile

- Total rows: 7059
- Total patients: 1013
- Default source sequence in config: T2
- Default experiment in config: A

## Subset counts
- train: 6251
- test: 808

## Group-level target counts
- P1 / test: 101 patients / 101 cases (T2+T2WI)
- P1 / train: 909 patients / 909 cases (T2+T2WI)
- P2 / test: 101 patients / 101 cases (DWI)
- P2 / train: 397 patients / 397 cases (DWI)
- P3 / train: 511 patients / 511 cases (ADC)
- P456 / test: 101 patients / 303 cases (T1+InPhase+OutPhase+C-pre)
- P456 / train: 909 patients / 1703 cases (T1+InPhase+OutPhase+C-pre)
- E_all / test: 101 patients / 303 cases (ARTERIAL+PORTAL+DELAY)
- E_all / train: 912 patients / 2731 cases (ARTERIAL+PORTAL+DELAY)

## Top raw-sequence source candidates
- DELAY: 911 patients / 911 cases
- ARTERIAL: 910 patients / 910 cases
- PORTAL: 910 patients / 910 cases
- T1: 512 patients / 512 cases
- T2: 512 patients / 512 cases (default)

## Experiment catalog
- A (dataset 311): train P1 only, test on P1/P2/P3/P456/E_all | train 909 patients / 909 cases | source T2+T2WI
- M1 (dataset 312): train on P1+P2+P3 | train 910 patients / 1817 cases | source T2+T2WI+DWI+ADC
- M2 (dataset 313): train on all plain MRI sequences | train 910 patients / 3520 cases | source T2+T2WI+DWI+ADC+T1+InPhase+OutPhase+C-pre
- M3 (dataset 314): train on all sequences | train 912 patients / 6251 cases | source T2+T2WI+DWI+ADC+T1+InPhase+OutPhase+C-pre+ARTERIAL+PORTAL+DELAY
- U1 (dataset 315): train on P2 only | train 397 patients / 397 cases | source DWI
- U2 (dataset 316): train on P3 only | train 511 patients / 511 cases | source ADC
- U3 (dataset 317): train on P456 only | train 909 patients / 1703 cases | source T1+InPhase+OutPhase+C-pre
- U4 (dataset 318): train on E_all only | train 912 patients / 2731 cases | source ARTERIAL+PORTAL+DELAY