# Misinformation Detection in Telegram Groups using GNNs

## Overview

This repository contains the codebase for the thesis project:
Classifying News Domains Shared in Telegram Chat Groups as
Misinformation using Graph Neural Networks (GNNs).

## Thesis Motivation

The spread of misinformation in online platforms, particularly
semi-anonymous messaging platforms like Telegram, poses a significant
societal challenge. This project models Telegram groups and shared URLs
as a graph structure, where:

-   Nodes represent domains grouped from URLs.
-   Edges represent sharing patterns in Telegram chat groups.

The hypothesis: Graph Neural Networks (GNNs) can outperform
state-of-the-art methods in classifying domains into misinformation by
leveraging graph structure.

## Repository Structure

### 01 Data Preprocessing & Feature Engineering

merge_URL_domain.ipynb Merges URLs to Lin et al. domains and
filters out social media domains.

scraper.py Scrapes article content from URLs using Selenium.

nlp_articles.py Generates semantic article embeddings using the
multilingual Sentence-BERT model
(paraphrase-multilingual-MiniLM-L12-v2).

train_test_split.ipynb Creates train/test splits and aggregates the
URL–chat bipartite matrix into a domain–chat representation.

feature_generation.ipynb Constructs node-level (domain-level) features,
including metadata and aggregated content features.

network_validation.ipynb Applies the Bipartite Configuration Model
(BiCM) to statistically validate domain similarities and construct the
validated monopartite projection.

topic_modeling_chat.ipynb Performs topic modeling on chat data using LDA
and Sentence-BERT to derive contextual chat-level features.

### 02 Baseline Models

random_classifier.ipynb Implements a random classifier as a chance-level
baseline.

MLP_content.ipynb Multi-Layer Perceptron (MLP) using content embeddings
and metadata.

MLP_content_agnostic.ipynb MLP using metadata-only features (no textual
embeddings).

### 03 Graph Neural Network Models

GCN_content_agnostic.ipynb Graph Neural Network models (GCN, GAT,
GraphSAGE) using only structural and metadata features.

GCN_content.ipynb GNN models combining network structure with article
embeddings.

## Project Workflow

### 1.  Preprocessing
    -   URL → Domain aggregation
    -   Article scraping
    -   Train/test split
    -   Network validation
### 2.  Feature Construction
    -   Metadata features
    -   Content embeddings
    -   Topic modeling
### 3.  Modeling
    -   Random baseline
    -   MLP baselines
    -   Graph Neural Networks
