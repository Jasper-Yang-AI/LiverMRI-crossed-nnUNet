# Experiment Suite

This file is auto-generated from configs/study_config.yaml.

## A

- dataset_id: 311
- source_bundle: P1
- source_members: T2, T2WI
- targets: P1, P2, P3, P456, E_all
- description: train P1 only, test on P1/P2/P3/P456/E_all
- run script: outputs\experiments\A\suite_commands\run_A.ps1

## M1

- dataset_id: 312
- source_bundle: M1
- source_members: T2, T2WI, DWI, ADC
- targets: P1, P2, P3, P456, E_all
- description: train on P1+P2+P3
- run script: outputs\experiments\M1\suite_commands\run_M1.ps1

## M2

- dataset_id: 313
- source_bundle: M2
- source_members: T2, T2WI, DWI, ADC, T1, InPhase, OutPhase, C-pre
- targets: P1, P2, P3, P456, E_all
- description: train on all plain MRI sequences
- run script: outputs\experiments\M2\suite_commands\run_M2.ps1

## M3

- dataset_id: 314
- source_bundle: M3
- source_members: T2, T2WI, DWI, ADC, T1, InPhase, OutPhase, C-pre, ARTERIAL, PORTAL, DELAY
- targets: P1, P2, P3, P456, E_all
- description: train on all sequences
- run script: outputs\experiments\M3\suite_commands\run_M3.ps1

## U1

- dataset_id: 315
- source_bundle: U1
- source_members: DWI
- targets: P1, P2, P3, P456, E_all
- description: train on P2 only
- run script: outputs\experiments\U1\suite_commands\run_U1.ps1

## U2

- dataset_id: 316
- source_bundle: U2
- source_members: ADC
- targets: P1, P2, P3, P456, E_all
- description: train on P3 only
- run script: outputs\experiments\U2\suite_commands\run_U2.ps1

## U3

- dataset_id: 317
- source_bundle: U3
- source_members: T1, InPhase, OutPhase, C-pre
- targets: P1, P2, P3, P456, E_all
- description: train on P456 only
- run script: outputs\experiments\U3\suite_commands\run_U3.ps1

## U4

- dataset_id: 318
- source_bundle: U4
- source_members: ARTERIAL, PORTAL, DELAY
- targets: P1, P2, P3, P456, E_all
- description: train on E_all only
- run script: outputs\experiments\U4\suite_commands\run_U4.ps1
