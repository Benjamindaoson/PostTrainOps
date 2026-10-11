## ADDED Requirements

### Requirement: Isolated supervised training
Training SHALL use GSM8K train and reject normalized question overlap with frozen
test. The audit MAY read frozen questions; labels SHALL NOT enter training.

#### Scenario: Overlap
- **WHEN** a normalized training question matches frozen test
- **THEN** preparation fails before creating usable training artifacts

### Requirement: Reproducible runs
Each run SHALL save explicit config, data hashes, package versions, environment,
timestamps, status, metrics and final checkpoint in a unique directory.

#### Scenario: Training failure
- **WHEN** training fails
- **THEN** the receipt records failure and no successful checkpoint is advertised

### Requirement: Fair paired evaluation
Base and SFT SHALL use the existing evaluator and identical frozen data, prompts,
tokenizer policy, decoding and seed. Comparison SHALL reject missing/duplicate IDs,
changed questions/gold, incompatible protocols and inconsistent metrics.

#### Scenario: Protocol mismatch
- **WHEN** evaluation inputs differ
- **THEN** comparison fails rather than publishing an improvement
