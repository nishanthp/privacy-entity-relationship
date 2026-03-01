## Privacy-Preserving NLP Experiments

A rigorous experimental framework for evaluating privacy-preserving NLP techniques using unbiased experimental methodology, real datasets, and statistical significance testing.

This project implements and evaluates a novel Context-Aware Entity Replacement approach and compares it against traditional privacy-preserving baselines.

The framework is designed to ensure fair, reproducible, and unbiased evaluation of privacy-preserving NLP techniques.

## Research Motivation

As large language models and conversational AI systems process sensitive text data, privacy-preserving transformations are required before model training or inference.

Traditional approaches such as:

 - Random replacement
 - Entity masking

Often either:

  - Destroy semantic meaning
  - Provide insufficient privacy protection

This project evaluates whether context-aware entity replacement can achieve:

 - Strong privacy protection
 - High semantic preservation
 - Statistical significance over baseline methods

## Key Features
Rigorous Experimental Design

The framework includes multiple bias-mitigation measures:

 - Randomized evaluation order
 - Balanced candidate pools
 - Multiple independent runs
 - Statistical significance testing
 - Fair dataset sampling

Each experiment is repeated 10 independent times to ensure reproducibility.

Privacy Methods Implemented

### Context-Aware Replacement

Uses sentence embeddings to select replacement entities that preserve semantic meaning.

Pipeline:

 - Named Entity Recognition
 - Context abstraction
 - Sentence embedding
 - Similarity-based candidate selection
 - Entity replacement

#### Advantages:

 - Maintains semantic meaning
 - Preserves sentence structure
 - Provides privacy protection

### Random Replacement (Baseline)

Randomly replaces entities with candidates from balanced pools.

#### Advantages:

 - Simple baseline
 - Unbiased comparison

#### Limitations:

 - Often damages semantic meaning

### Entity Masking (Baseline)

Replaces entities with type tokens.

Example:

  Original:
    Alex Johnson works at Alpha Corp in Riverside.

  Masked:
    [PERSON] works at [ORG] in [GPE].

#### Advantages:
 - Strong privacy protection

#### Limitations:

 - Loses semantic information

### Original Text (Control)

No privacy transformation.

Used as experimental control.
