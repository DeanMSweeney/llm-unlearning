# Machine Unlearning for Mitigating Social Biases in LLMs

Large language models (LLMs) inherit and amplify social biases present in their training data. This motivates the need for targeted debiasing methods that preserve model utility. In this repository, we reproduce the Partitioned Contrastive Gradient Unlearning (PCGU) method and extend it from binary gender bias to a multi-class setting that includes neutral demographic categories. We introduce a spread-based gradient importance metric to identify and selectively update parameters most responsible for multi-class bias. Experiments on BERT and RoBERTa show that the proposed multi-class PCGU approach achieves bias scores closest to the unbiased target while maintaining substantially higher language quality than existing debiasing methods.

This work was completed as a class project for EECS 598 as an extension of the work described in Yu, C., Jeoung, S., Kasi, A., Yu, P., & Ji, H. (2023), *Findings of the ACL*.

![Poster](poster.png)

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

- `Binary/` - Binary classification implementation for bias unlearning
  - `train.py` - Training entry point with gradient-based unlearning
  - `eval.py` - StereoSet evaluation pipeline
  - `eval_crows.py` - CrowS-Pairs evaluation
  - `utils/` - Trainer, dataset builders, and utility functions
  - `logs/` - Evaluation data
- `MultiClass/` - Multi-class classification implementation
  - `train.py` - Training pipeline
  - `eval.py` - Evaluation scripts
  - `utils/` - Supporting utilities
  - `logs/` - Evaluation data
- `data/` - Datasets for training and evaluation

## Original Work

This is a recreation of methods from the original PCGU-UnlearningBias repository. For questions, implementation details, or the original research, please refer to:

https://github.com/CharlesYu2000/PCGU-UnlearningBias

## Acknowledgments

This project was completed as part of EECS 598 coursework, implementing and validating the gradient-based unlearning approach for bias mitigation in language models.
