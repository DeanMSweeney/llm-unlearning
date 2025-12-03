# LLM Bias Unlearning

This repository contains an implementation of gradient-based unlearning techniques for mitigating gender bias in pre-trained language models. This work was completed as a class project for EECS 598, recreating the methods described in the PCGU (Parameter-Efficient Contrastive Gradient Unlearning) approach.

## Overview

The project implements a selective parameter update strategy to reduce gender bias in masked language models (MLMs) such as BERT and RoBERTA. The approach:

1. **Training/Unlearning**: Uses the WinoGender dataset to identify and mitigate gender bias by selectively updating model parameters through gradient-based optimization
2. **Evaluation**: Measures bias reduction using the StereoSet benchmark across multiple domains (gender, profession, race, religion)

Key features:
- Top-k parameter selection for efficient, targeted unlearning
- Flexible gradient aggregation strategies (input vs. output layer)
- Dynamic or static gradient direction selection (advantaged vs. disadvantaged groups)
- Comprehensive evaluation metrics including Stereotype Score (SS), Language Modeling Score (LMS), and ICAT

## Project Structure

- `Gender/` - Main implementation for gender bias unlearning
  - `main.py` - Training entry point with gradient-based unlearning
  - `eval.py` - StereoSet evaluation pipeline
  - `trainer.py` - Unbias trainer implementation
  - `utils/` - Dataset builders and utility functions

## Original Work

This is a recreation of methods from the original PCGU-UnlearningBias repository. For questions, implementation details, or the original research, please refer to:

https://github.com/CharlesYu2000/PCGU-UnlearningBias

## Acknowledgments

This project was completed as part of EECS 598 coursework, implementing and validating the gradient-based unlearning approach for bias mitigation in language models.
