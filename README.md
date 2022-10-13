# Optimizing Warfarin Dosing for Patients with Atrial Fibrillation Using Deep Reinforcement Learning

See [REPLICATION](REPLICATION.md) for details on how to replicate the study on COMBINE-AF data.

## System requirements

The software has been tested on Ubuntu Linux 20.04, Python 3.9 and PyTorch 1.9, using GPUs. GPUs are not required, but are recommended for performance reasons.

## Installation & demo

Please see [here](https://github.com/hamilton-health-sciences/smdbcq) for instructions on installing the PyTorch model and running it on synthetic CartPole data from D3RLPY. Installation should take less than a minute.

## Citation

> Optimizing Warfarin Dosing for Patients with Atrial Fibrillation Using Deep Reinforcement Learning
>
> Jeremy Petch, Walter Nelson, Mary Wu, Marzyeh Ghassemi, Alexander Benz, Mehdi Fatemi, Shuang Di, Anthony Carnicelli, Christopher Granger, Robert Giugliano, Hwanhee Hong, Manesh Patel, Lars Wallentin, John Eikelboom, Stuart J. Connolly
>
> *Under review.*

## See also

Our earlier methodology work:

> [Semi-Markov Offline Reinforcement Learning for Healthcare](https://arxiv.org/abs/2203.09365)
>
> Mehdi Fatemi, Mary Wu, Jeremy Petch, Walter Nelson, Stuart J. Connolly, Alexander Benz, Anthony Carnicelli, Marzyeh Ghassemi
>
> Conference on Health, Inference, and Learning 2022

The paper describing the COMBINE-AF data:

> [Individual Patient Data from the Pivotal Randomized Controlled Trials of Non-Vitamin K Antagonist Oral Anticoagulants in Patients with Atrial Fibrillation (COMBINE AF): Design and Rationale](https://pubmed.ncbi.nlm.nih.gov/33296688/)
>
> Anthony P Carnicelli, Hwanhee Hong, Robert P Giugliano, Stuart J Connolly, John Eikelboom, Manesh R Patel, Lars Wallentin, David A Morrow, Daniel Wojdyla, Kaiyuan Hua, Stefan H Hohnloser, Jonas Oldgren, Christian T Ruff, Jonathan P Piccini, Renato D Lopes, John H Alexander, Christopher B Granger, COMBINE AF Investigators
>
> American Heart Journal
