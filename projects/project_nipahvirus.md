---
layout: project
title: Nipah Virus Binder Design
subtitle: PhD Student • Noelia Ferruz Lab • 2025
---

## Introduction

This project addresses the challenge of designing binders for the Nipah virus. More details about the competition can be found [here](https://proteinbase.com/competitions/adaptyv-nipah-competition).

## Methods

**ProtRL** was applied to a fine-tuned protein language model (pLM) using **epirin B** as the starting point. The **REINFORCE** algorithm was utilized with a comprehensive reward function that included:

- Length
- PAE (Predicted Aligned Error)
- Shape Complementarity
- LIS (Local Interaction Score)
- IPTM_D0chn
- dRMSD
- IPSAE
- Number of clusters

The first selection step was based on the implicit reward, followed by rigorous in-silico metrics to identify the most promising candidates.

## Other Strategies

In addition to the main approach, several other strategies were tested:

- Nanobody fine-tuning with the same aim, starting with nanobody **n424**.
- Using **REINFORCE** with different configurations.
- Using **GRPO** (Group Relative Policy Optimization).
- Testing **CoT** (Chain of Thought) - *soon to come*.
