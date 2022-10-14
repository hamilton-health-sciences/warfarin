# Optimizing Warfarin Dosing for Patients with Atrial Fibrillation Using Deep Reinforcement Learning

## Software

In addition to this repository, which contains all the software needed to reproduce our results exactly, we have published a Pip package containing the semi-markov discrete BCQ reinforcement learning PyTorch model. Instructions for installing and using this package on synthetic data are available [here](https://github.com/hamilton-health-sciences/smdbcq). Briefly:

* Requirements: The model has been tested on Ubuntu Linux 20.04, Python 3.9, and GPUs (although GPUs are not required).
* Installation: `python3 -m pip install smdbcq`
* Demo: `python3 -m smdbcq --demo` to run on CartPole data.
* Installation time should be < 1 minute on any machine.
* Runtime should be < 1 second per optimization step, depending on hardware, with hundreds of thousands to millions of steps required for robust model estimation. We would expect replicating our study with 4 Tesla V100 GPUs to take approximately one week of computation time.
* Please see [REPLICATION](REPLICATION.md) for instructions to run the software on our data and detailed replication instructions.

## Citation

> Optimizing Warfarin Dosing for Patients with Atrial Fibrillation Using Deep Reinforcement Learning
>
> Jeremy Petch, Walter Nelson, Mary Wu, Marzyeh Ghassemi, Alexander Benz, Mehdi Fatemi, Shuang Di, Anthony Carnicelli, Christopher Granger, Robert Giugliano, Hwanhee Hong, Manesh Patel, Lars Wallentin, John Eikelboom, Stuart J Connolly
>
> *Under review.*

## See also

Our earlier methodology work:

> [Semi-Markov Offline Reinforcement Learning for Healthcare](https://arxiv.org/abs/2203.09365)
>
> Mehdi Fatemi, Mary Wu, Jeremy Petch, Walter Nelson, Stuart J Connolly, Alexander Benz, Anthony Carnicelli, Marzyeh Ghassemi
>
> Conference on Health, Inference, and Learning 2022

The paper describing the COMBINE-AF data:

> [Individual Patient Data from the Pivotal Randomized Controlled Trials of Non-Vitamin K Antagonist Oral Anticoagulants in Patients with Atrial Fibrillation (COMBINE AF): Design and Rationale](https://pubmed.ncbi.nlm.nih.gov/33296688/)
>
> Anthony P Carnicelli, Hwanhee Hong, Robert P Giugliano, Stuart J Connolly, John Eikelboom, Manesh R Patel, Lars Wallentin, David A Morrow, Daniel Wojdyla, Kaiyuan Hua, Stefan H Hohnloser, Jonas Oldgren, Christian T Ruff, Jonathan P Piccini, Renato D Lopes, John H Alexander, Christopher B Granger, COMBINE AF Investigators
>
> American Heart Journal

The original discrete batch-constrained Q-learning paper, upon which the semi-Markov form of the model is based:

> [Benchmarking Batch Deep Reinforcement Learning Algorithms](https://arxiv.org/abs/1910.01708)
>
> Scott Fujimoto, Edoardo Conti, Mohammad Ghavamzadeh, Joelle Pineau
>
> arXiv preprint
