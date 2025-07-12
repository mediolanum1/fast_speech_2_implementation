# Implementation of Fast Speech 2 TTS Model

<p align="center">
  <a href="#about">About</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#training-results">Training results</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository provides an implementation of FastSpeech 2 using PyTorch, a non-autoregressive text-to-speech (TTS) model that improves prosody modeling by incorporating duration, pitch, and energy predictors.

## Dataset

LJSpeech dataset that was fully processed and besides original LJSpeech's audios and transcripts also contains textGrid files with extracted phonemes and their durations for each wav in transcripts folder, as well as extracted mel specs, energy levels and pitch for each audio was created and uploaded to Kaggle [`LJSpeech processed`](https://www.kaggle.com/datasets/etozherobert/ljspeech-processed)



