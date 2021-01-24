# Kronos: Self-Supervised Temporal Metric Learning

Kronos is a PyTorch library for learning and evaluating self-supervised representations learned on video data.

## Models

- [x] Temporal Cycle Consistency
- [ ] Time Contrastive Networks
    - [x] Single-view TCN
    - [ ] Multi-view TCN
- [x] Shuffle and Learn
- [ ] Temporal Distance Classification

## Datasets

- [x] MIME
- [x] Penn Action
- [ ] Pouring (seems to be unavailable at the moment)

## Evaluators

- Requires labels
    - [x] Phase alignment
        - [x] Topk
        - [ ] Error
- Does not require labels
    - [x] Frame Nearest Neighbour Visualizer
    - [x] Kendall's Tau
    - [x] Cycle-consistency (2-way and 3-way)

## Todos

- [ ] Add pretrained models with `hubconf`.
- [ ] Polish download scripts for all three datasets.
- [ ] Add mixed precision to checkpointing.
- [ ] Add distributed training.
- [ ] Add lr scheduler factory.
