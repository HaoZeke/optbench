# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Org-mode source, exported to reStructuredText with Emacs, then built with Sphinx and the Shibuya theme. The same path as the rgpot manual. Comparison tables are generated from the oracle. Astro was left aside.

## Users

A person comparing geometry optimizers, saddle searches, or global optimization, usually with a named problem in front of them and a force-call budget in mind.

## Product Purpose

OptBench publishes the cost of a finished answer on a fixed set of problems. Success is a reader finding the stopping rule and the ranked methods for one problem without opening the archive.

## Positioning

The rank is the archived `benchmark.dat`, and the stopping rule is the problem file. A one-start run is labeled short and is not a row in that rank.

## Operating Context

Problems and scores live in this repository. The public site is the rendering of those files. A full campaign runs on a cluster and writes another `benchmark.dat`. MATLAB checks run as a separate two-search batch. Code tarballs from the old website archive stay in that archive.

## Capabilities and Constraints

The site shows every archived score from commit `0f26b796` of the public archive, including rows the old site marked held out. It does not claim a new campaign has been run, and it does not display the old code tarballs.

## Brand Commitments

The name is OptBench. The citation is Chill, Stevenson, Ruehle, Shang, Xiao, Farrell, Wales, and Henkelman, J. Chem. Theory Comput. 2014, 10, 5476-5482, doi:10.1021/ct5008718.

## Evidence on Hand

`problems/` holds the contracts. `oracle/entries.jsonl` holds the 151 scores. `softsaddle/entry.toml` holds the April 2019 SoftSaddle score and the two-search batch.

## Product Principles

The number on the page is a field from `benchmark.dat`.
The stopping rule is visible on the same page as the rank.
A shared contract says so.
A two-search batch is not mixed into the archived rank.
The old archive remains the provenance of the scores.
